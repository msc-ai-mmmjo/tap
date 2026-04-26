"""
Inference latency / throughput benchmark for vanilla OLMo, naive-averaging
Hydra, and Hydra + PoE (Product of Experts speculative decode).

Three configurations are timed back-to-back on the same GPU so the numbers are
directly comparable:

1. ``baseline``    — vanilla OLMo (1B or 7B), random weights. Fastest per
   token; sets the upper bound any Hydra variant can approach.
2. ``hydra_naive`` — HydraTransformer run with all heads in series and logits
   averaged (the default ``forward_and_sample`` codepath). Random weights.
   Strictly more compute than (1) — every step pays trunk + N head forwards.
3. ``hydra_poe``   — Hydra wrapped in :class:`olmo_tap.inference.poe.PoE`,
   running γ-step speculative decode with one draft head + (N-1)-head verifier
   jury (Product of Experts: log-probs are summed across verifier heads,
   i.e. multiplied in probability space). Real merged-LoRA weights via
   :func:`olmo_tap.inference.loading_weights.load_ensemble` so acceptance
   rate is meaningful. Optionally swept across γ values.

Knobs live in ``config.json`` next to this file. The ``baseline`` and
``hydra_naive`` rows use random weights because per-step latency is
architecture-bound — only PoE needs real weights (acceptance rate depends on
the actual model behavior).

Usage::

    pixi run -e cuda python -m olmo_tap.benchmarks.inference

Output lands in ``olmo_tap/benchmarks/results/<YYYY-MM-DD>_run<NN>/``:

* ``results.json`` — raw timings + per-position decode stats + PoE per-γ stats.
* ``graph.png``    — TTFT KDE, per-position decode latency, per-position TPS.

NOTE:
PoE timing reports a single ``call_median_ms`` per γ — the wall time of one
full ``generate_with_cache`` call (max_new_tokens tokens). The
``effective_tps`` field is ``1000 * accepted_tokens / call_median_ms``, i.e.
draft-acceptance throughput. Under non-zero resample rate the user-visible
TPS is slightly higher because resampled positions are still real output
tokens; see :func:`benchmark_poe` docstring for the exact definitions.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import cast, Any

import torch

from olmo_core.nn.attention import AttentionBackendName, Attention, KVCacheManager
from olmo_core.nn.transformer import Transformer, TransformerBlock
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_tap.constants import VOCAB_SIZE
from olmo_tap.hydra import HydraTransformer, HydraTransformerConfig


def build_hydra_model(
    n_heads=5, heads_depth=3, dtype=torch.bfloat16, device="cuda", size="1b"
):
    """Build a HydraTransformer for naive-averaging benchmarks (random weights).

    :param n_heads: Number of parallel heads in the Hydra branching.
    :param heads_depth: Layers per head (trunk gets ``n_layers - heads_depth``).
    :param dtype: Compute dtype.
    :param device: Target device.
    :param size: ``"1b"`` or ``"7b"`` — picks the OLMo factory.
    :returns: A built, ``.eval()``-mode HydraTransformer with random weights.

    NOTE: weights are random — latency is architecture-bound at this
    granularity. Only the PoE row needs real weights (for acceptance-rate
    realism); see :func:`build_poe`.
    """
    if size == "7b":
        config = HydraTransformerConfig.from_olmo2_7B(
            n_heads=n_heads, heads_depth=heads_depth
        )
    else:
        config = HydraTransformerConfig.from_olmo2_1B(
            n_heads=n_heads, heads_depth=heads_depth
        )
    model = config.build(init_device="cpu")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def init_kv_cache(model, batch_size, max_seq_len):
    if isinstance(model, HydraTransformer):
        model.init_kv_cache(batch_size, max_seq_len)
    else:
        for block in model.blocks.values():
            block.attention.init_kv_cache_manager(batch_size, max_seq_len)


def build_baseline_model(dtype=torch.bfloat16, device="cuda", size="1b"):
    """Build a vanilla single-tower OLMo Transformer (random weights).

    :param dtype: Compute dtype.
    :param device: Target device.
    :param size: ``"1b"`` or ``"7b"``.
    :returns: A built, ``.eval()``-mode Transformer using the FlashAttention-2
        backend, with random weights (latency is architecture-bound).
    """
    if size == "7b":
        config = TransformerConfig.olmo2_7B(vocab_size=VOCAB_SIZE)
    else:
        config = TransformerConfig.olmo2_1B_v2(vocab_size=VOCAB_SIZE)
    config.block.sequence_mixer.backend = AttentionBackendName.flash_2  # type: ignore
    model = config.build(init_device="cpu")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def build_poe(cfg, dtype=torch.bfloat16):
    """Build a :class:`PoE` (Product of Experts speculative decoder) with real
    merged-LoRA weights from :func:`load_ensemble`.

    Unlike the baseline / naive-Hydra rows (which use random weights), PoE is
    built on the deployed 7B Hydra with security + robustness LoRAs merged per
    head plus the uncertainty head, because PoE's acceptance rate depends on
    the actual logit distributions the verifier heads produce. Random weights
    would give meaningless accept/reject behavior.

    :param cfg: Parsed ``config.json`` dict. Reads ``poe_gamma`` (overridden
        per-call when sweeping), ``poe_beta``, ``poe_max_new_tokens``.
    :param dtype: Compute dtype to cast the loaded ensemble to.
    :returns: ``(poe, model)`` — the :class:`PoE` wrapper plus the underlying
        :class:`HydraTransformer` (returned so the caller can ``del`` it for
        VRAM cleanup before the next config).
    """
    from olmo_tap.constants import WEIGHTS_DIR
    from olmo_tap.inference.loading_weights import load_ensemble
    from olmo_tap.inference.poe import PoE
    from transformers import AutoTokenizer

    model, n_heads = load_ensemble()
    model.to(dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    poe = PoE(
        model=model,
        tokenizer=tokenizer,
        n_llm_heads=n_heads - 1,  # last head is uncertainty
        gamma=cfg["poe_gamma"],
        beta=cfg["poe_beta"],
        max_new_tokens=cfg["poe_max_new_tokens"],
    )
    return poe, model


def get_all_kv_cache_managers(model) -> list[KVCacheManager | None]:
    managers = []
    if isinstance(model, HydraTransformer):
        for block in model.trunk.blocks.values():
            attn = cast(TransformerBlock, block).attention
            if isinstance(attn, Attention):
                managers.append(attn.kv_cache_manager)
        for head in model.heads:
            for block in cast(Transformer, head).blocks.values():
                attn = cast(TransformerBlock, block).attention
                if isinstance(attn, Attention):
                    managers.append(attn.kv_cache_manager)
    else:
        for block in cast(Transformer, model).blocks.values():
            attn = cast(TransformerBlock, block).attention
            if isinstance(attn, Attention):
                managers.append(attn.kv_cache_manager)

    return managers


def reset_kv_cache_position(managers: list[KVCacheManager | None], position):
    for m in managers:
        if m is not None:
            m.cache_seqlens.fill_(position)


def forward_and_sample(model, input_ids):
    logits = model(input_ids)
    if isinstance(model, HydraTransformer):
        return logits[:, 0, -1, :].mean(dim=0).argmax()
    else:
        return logits[0, -1, :].argmax()


def benchmark_ttft(model, prompt_ids, warmup_ms=100.0, rep_ms=1000.0):
    from olmo_tap.benchmarks.harness import (
        benchmark,
        compute_stats,
        filter_outliers_iqr,
    )

    managers = get_all_kv_cache_managers(model)

    def setup():
        reset_kv_cache_position(managers, 0)

    def fn():
        forward_and_sample(model, prompt_ids)

    raw = benchmark(fn, warmup_ms, rep_ms, flush_l2=True, setup=setup)
    filtered = filter_outliers_iqr(raw)

    return {
        "raw_ms": raw,
        "filtered_ms": filtered,
        **compute_stats(filtered),
    }


def benchmark_decode(
    model, prompt_ids, gen_length=128, step_interval=8, warmup_ms=100.0, rep_ms=1000.0
):
    from olmo_tap.benchmarks.harness import (
        benchmark,
        compute_stats,
        filter_outliers_iqr,
    )

    managers = get_all_kv_cache_managers(model)
    positions = list(range(0, gen_length, step_interval))
    results = {"positions": positions, "per_position": {}}

    if not managers or managers[0] is None:
        return results  # TODO: better error handling logic

    for m in managers:
        if m is not None:
            m.zero_cache()
    last_token = forward_and_sample(model, prompt_ids).unsqueeze(0).unsqueeze(0)

    for step in range(gen_length):
        if step in positions:
            saved_pos = managers[0].cache_seqlens.item()

            def setup(pos=saved_pos):
                reset_kv_cache_position(managers, pos)

            def fn(tok=last_token):
                forward_and_sample(model, tok)

            raw = benchmark(fn, warmup_ms, rep_ms, flush_l2=True, setup=setup)
            filtered = filter_outliers_iqr(raw)
            stats = compute_stats(filtered)
            stats["tps"] = round(1000.0 / stats["median_ms"], 2)

            results["per_position"][str(saved_pos)] = {
                "raw_ms": raw,
                "filtered_ms": filtered,
                **stats,
            }

            reset_kv_cache_position(managers, saved_pos)

        last_token = forward_and_sample(model, last_token).unsqueeze(0).unsqueeze(0)

    return results


def benchmark_poe(poe, prompt_text, warmup_ms=100.0, rep_ms=1000.0):
    """Time end-to-end :meth:`PoE.generate_with_cache` and compute throughput
    + acceptance statistics.

    PoE is timed at call granularity (one call = ``poe.max_new_tokens`` output
    tokens), not per step, because the γ-step draft loop and the verifier
    pass don't decompose into a single repeating "fn" the way naive decode
    does. :meth:`PoE.generate_with_cache` re-inits its own KV cache on every
    call, so the harness ``setup`` callback is left unset.

    The reported metrics for a benchmark window:

    * ``median_ms``                       — median wall time of one full call.
    * ``avg_accepted_tokens_per_call``    — mean over calls of
      ``output_tokens − resampled_tokens`` (tokens the draft head got right).
    * ``resample_rate``                   — total resampled / total output
      tokens; the fraction of output positions where the verifier jury
      rejected the draft and a corrected token was sampled. Lower is better.
    * ``effective_tps``                   — ``1000 * avg_accepted / median_ms``;
      **draft-acceptance throughput**, not user-visible throughput. Under
      non-zero ``resample_rate`` user-visible TPS is slightly higher because
      resampled positions are still real output tokens; the difference is
      ``resample_rate * effective_tps / (1 - resample_rate)``.
    * ``n_calls``                         — number of timed full-generation
      calls (estimate + warmup + measurement).

    :param poe: :class:`PoE` instance. ``poe.gamma`` may be mutated by the
        caller between invocations to sweep γ; everything else stays put.
    :param prompt_text: A single-turn user message string. Will be wrapped
        with ``apply_chat_template`` inside :meth:`generate_with_cache`.
    :param warmup_ms: Warmup budget passed to :func:`harness.benchmark`.
    :param rep_ms: Measurement budget. Should comfortably exceed one call's
        wall time so multiple measurements land — a 64-token PoE call on 7B
        is ~3-5 s, so use ``rep_ms >= 8000`` for stable medians.
    :returns: Stats dict with the keys described above plus ``raw_ms`` and
        ``filtered_ms`` (post-IQR-filtered timings).
    """
    from olmo_tap.benchmarks.harness import (
        benchmark,
        compute_stats,
        filter_outliers_iqr,
    )

    accepted_total = [0]
    resampled_total = [0]
    calls = [0]

    def fn():
        output_parts, _orig, resampled_idxs, _p = poe.generate_with_cache(
            prompt_text, is_mcq=False, temperature=None
        )
        n_tokens = len(output_parts) - 1
        accepted_total[0] += n_tokens - len(resampled_idxs)
        resampled_total[0] += len(resampled_idxs)
        calls[0] += 1

    raw = benchmark(fn, warmup_ms, rep_ms, flush_l2=True)
    filtered = filter_outliers_iqr(raw)
    stats = compute_stats(filtered)

    n_calls = max(calls[0], 1)
    avg_accepted = accepted_total[0] / n_calls
    total_tokens = accepted_total[0] + resampled_total[0]
    resample_rate = resampled_total[0] / total_tokens if total_tokens > 0 else 0.0
    effective_tps = (
        round(1000.0 * avg_accepted / stats["median_ms"], 2)
        if stats["median_ms"] > 0
        else 0.0
    )

    return {
        "raw_ms": raw,
        "filtered_ms": filtered,
        **stats,
        "avg_accepted_tokens_per_call": round(avg_accepted, 2),
        "resample_rate": round(resample_rate, 4),
        "effective_tps": effective_tps,
        "n_calls": n_calls,
    }


def make_output_dir():
    base = Path(__file__).parent / "results"
    base.mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    # find existing runs for today, auto-increment
    existing = [d.name for d in base.iterdir() if d.name.startswith(today)]
    run_nums = [
        int(match.group(1))
        for name in existing
        if (match := re.search(r"run(\d+)", name)) is not None
    ]
    next_run = max(run_nums, default=0) + 1

    out = base / f"{today}_run{next_run:02d}"
    out.mkdir()
    return out


def load_config():
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        return json.load(f)


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def run_benchmarks(model, prompt_ids, cfg, label):
    print(f"\n--- {label} ---")

    print("Benchmarking TTFT...")
    ttft = benchmark_ttft(model, prompt_ids, cfg["warmup_ms"], cfg["rep_ms"])
    print(f"  TTFT median: {ttft['median_ms']:.2f} ms")

    print("Benchmarking decode...")
    decode = benchmark_decode(
        model,
        prompt_ids,
        cfg["generation_length"],
        cfg["decode_step_interval"],
        cfg["warmup_ms"],
        cfg["rep_ms"],
    )
    tps_values = [v["tps"] for v in decode["per_position"].values()]  # type: ignore
    avg_tps = sum(tps_values) / len(tps_values)
    print(f"  Avg TPS: {avg_tps:.1f} tokens/sec")

    return {"ttft": ttft, "decode": decode}


def main():
    cfg = load_config()
    dtype = DTYPE_MAP[cfg["dtype"]]
    size = cfg.get("size", "1b")
    max_seq_len = cfg["prompt_length"] + cfg["generation_length"]

    torch.manual_seed(42)
    prompt_ids = torch.randint(0, VOCAB_SIZE, (1, cfg["prompt_length"]), device="cuda")

    results: dict[str, Any] = {}

    if cfg.get("baseline", False):
        print(f"\nBuilding baseline Transformer ({size})...")
        baseline = build_baseline_model(dtype, size=size)
        init_kv_cache(baseline, batch_size=1, max_seq_len=max_seq_len)

        with torch.no_grad():
            results["baseline"] = run_benchmarks(
                baseline, prompt_ids, cfg, f"Baseline ({size})"
            )
        del baseline
        torch.cuda.empty_cache()

    print(f"\nBuilding HydraTransformer ({size}, naive averaging)...")
    model = build_hydra_model(cfg["n_heads"], cfg["heads_depth"], dtype, size=size)
    init_kv_cache(model, batch_size=1, max_seq_len=max_seq_len)

    with torch.no_grad():
        results["hydra_naive"] = run_benchmarks(
            model, prompt_ids, cfg, f"Hydra ({size}, naive avg)"
        )
    del model
    torch.cuda.empty_cache()

    if cfg.get("poe", False):
        print("\nBuilding PoE ensemble (real merged-LoRA weights)...")
        poe, poe_model = build_poe(cfg, dtype=dtype)

        gammas = cfg.get("poe_gammas") or [cfg["poe_gamma"]]
        rep_ms_poe = cfg.get("poe_rep_ms", cfg["rep_ms"])
        per_gamma: dict[str, Any] = {}
        with torch.no_grad():
            for gamma in gammas:
                poe.gamma = gamma
                print(f"\n--- Hydra + PoE (gamma={gamma}) ---")
                stats = benchmark_poe(
                    poe,
                    cfg["poe_prompt"],
                    warmup_ms=cfg["warmup_ms"],
                    rep_ms=rep_ms_poe,
                )
                print(
                    f"  call median: {stats['median_ms']:.1f} ms · "
                    f"avg accepted: {stats['avg_accepted_tokens_per_call']} tok · "
                    f"effective TPS: {stats['effective_tps']} · "
                    f"resample: {stats['resample_rate']} · n_calls={stats['n_calls']}"
                )
                per_gamma[str(gamma)] = stats
        results["hydra_poe"] = {"per_gamma": per_gamma}
        del poe, poe_model
        torch.cuda.empty_cache()

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        **{
            k: cfg[k]
            for k in [
                "size",
                "dtype",
                "n_heads",
                "heads_depth",
                "prompt_length",
                "generation_length",
                "decode_step_interval",
            ]
            if k in cfg
        },
        "poe_gamma": cfg.get("poe_gamma"),
        "poe_beta": cfg.get("poe_beta"),
        "poe_max_new_tokens": cfg.get("poe_max_new_tokens"),
    }
    results["metadata"] = cast(Any, metadata)

    out_dir = make_output_dir()
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    from olmo_tap.benchmarks.plotting import plot_results

    plot_results(results, out_dir)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()

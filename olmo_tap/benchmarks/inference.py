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


def build_hydra_model(n_heads=5, heads_depth=3, dtype=torch.bfloat16, device="cuda"):
    # NOTE: attention backend handled by HydraTransformerConfig.from_olmo2_1B
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


def build_baseline_model(dtype=torch.bfloat16, device="cuda"):
    config = TransformerConfig.olmo2_1B_v2(vocab_size=VOCAB_SIZE)
    config.block.sequence_mixer.backend = AttentionBackendName.flash_2  # type: ignore
    model = config.build(init_device="cpu")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


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
    max_seq_len = cfg["prompt_length"] + cfg["generation_length"]

    torch.manual_seed(42)
    prompt_ids = torch.randint(0, VOCAB_SIZE, (1, cfg["prompt_length"]), device="cuda")

    print("Building HydraTransformer...")
    model = build_hydra_model(cfg["n_heads"], cfg["heads_depth"], dtype)
    init_kv_cache(model, batch_size=1, max_seq_len=max_seq_len)

    with torch.no_grad():
        results = {"hydra": run_benchmarks(model, prompt_ids, cfg, "HydraTransformer")}

    if cfg["baseline"]:
        del model
        torch.cuda.empty_cache()
        print("\nBuilding baseline Transformer...")
        baseline = build_baseline_model(dtype)
        init_kv_cache(baseline, batch_size=1, max_seq_len=max_seq_len)

        with torch.no_grad():
            results["baseline"] = run_benchmarks(
                baseline, prompt_ids, cfg, "Baseline (16-layer)"
            )

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        **{
            k: cfg[k]
            for k in [
                "dtype",
                "n_heads",
                "heads_depth",
                "prompt_length",
                "generation_length",
                "decode_step_interval",
            ]
        },
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

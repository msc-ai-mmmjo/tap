"""Entrant -> response generation for the configuration-level Elo tournament.

Each :class:`EntrantSpec` is materialised on GPU via
:func:`build_entrant`, then fed the prompt bank one prompt at a time.
The vanilla-HF entrant uses greedy decoding directly through
``model.generate``; the Hydra entrants route through
:meth:`PoE.generate_with_cache` with per-prompt seeding so the random
draft-head selection lines up across the three Hydra entrants on each
prompt while still varying across prompts.

Responses are persisted line-by-line to
``caches/responses/responses_<entrant_id>.jsonl`` so partial runs are
resumable: re-running the script picks up only the cache misses.

Usage::

    pixi run -e cuda python -m olmo_tap.final_evals.elo.generate \\
        --bank olmo_tap/final_evals/elo/prompts/bank.jsonl \\
        --entrants base_olmo,security_only,security_plus_robustness,full_poe \\
        --cache-dir olmo_tap/final_evals/elo/caches/responses
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch

from olmo_tap.experiments.utils.random_seed import set_seed
from olmo_tap.final_evals.elo.entrants import (
    ENTRANTS,
    ENTRANTS_BY_ID,
    EntrantSpec,
    LoadedEntrant,
    build_entrant,
)
from olmo_tap.final_evals.elo.types import (
    GeneratedResponse,
    Prompt,
    load_prompt_bank,
    prompt_seed,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cache_path(cache_dir: Path, entrant_id: str) -> Path:
    return cache_dir / f"responses_{entrant_id}.jsonl"


def _load_response_cache(
    cache_dir: Path, entrant_id: str
) -> dict[str, GeneratedResponse]:
    """Read the existing cache for one entrant, keyed by prompt_id."""
    path = _cache_path(cache_dir, entrant_id)
    cached: dict[str, GeneratedResponse] = {}
    if not path.exists():
        return cached
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cached[entry["prompt_id"]] = GeneratedResponse(**entry)
    return cached


def _append_response(
    cache_dir: Path, entrant_id: str, record: GeneratedResponse
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, entrant_id)
    with path.open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")
        f.flush()


def _strip_trailing_eos(parts: list[str], eos_str: str) -> str:
    """Drop trailing EOS tokens emitted by PoE's tokenwise decode."""
    cleaned = list(parts)
    while cleaned and eos_str and cleaned[-1] == eos_str:
        cleaned.pop()
    return "".join(cleaned)


def _generate_vanilla_hf(
    spec: EntrantSpec,
    loaded: LoadedEntrant,
    prompt: Prompt,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """Greedy generation through a vanilla HuggingFace causal LM."""
    assert loaded.hf_model is not None
    tok = loaded.tokenizer
    # apply_chat_template with tokenize+return_tensors returns a BatchEncoding
    # (dict-like) rather than a bare Tensor; pull the input_ids tensor out
    # explicitly so .size() / slicing works.
    enc = cast(
        dict[str, torch.Tensor],
        tok.apply_chat_template(
            [{"role": "user", "content": prompt.text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ),
    )
    input_ids = enc["input_ids"].to("cuda")
    n_input = input_ids.size(1)

    out = loaded.hf_model.generate(
        input_ids=input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    new_tokens = out[0, n_input:]
    response_text = cast(str, tok.decode(new_tokens, skip_special_tokens=True))
    diagnostics: dict[str, Any] = {
        "draft_head_idx": None,
        "seed": prompt_seed(prompt.prompt_id),
        "bypass_jury": None,
        "n_resampled": 0,
        "n_tokens_generated": int(new_tokens.size(0)),
    }
    return response_text, diagnostics


def _generate_custom_poe(
    spec: EntrantSpec,
    loaded: LoadedEntrant,
    prompt: Prompt,
) -> tuple[str, float | None, dict[str, Any]]:
    """PoE generation, routed through the eval-mode kwargs on
    :meth:`generate_with_cache`."""
    assert loaded.poe is not None
    parts, _orig, _resampled, p_correct = loaded.poe.generate_with_cache(
        prompt_text=prompt.text,
        compute_uncertainty=spec.needs_uncertainty,
        seed=prompt_seed(prompt.prompt_id),
        bypass_jury=spec.bypass_jury,
        temperature=spec.temperature,
    )
    eos_id = loaded.tokenizer.eos_token_id
    eos_str = cast(str, loaded.tokenizer.decode([eos_id])) if eos_id is not None else ""
    response_text = _strip_trailing_eos(list(parts[1:]), eos_str)
    diagnostics = dict(loaded.poe.last_diagnostics)
    return response_text, p_correct, diagnostics


def generate_responses_for_entrant(
    spec: EntrantSpec,
    loaded: LoadedEntrant,
    prompts: list[Prompt],
    cache_dir: Path,
    max_new_tokens: int = 256,
) -> list[GeneratedResponse]:
    """Generate (or recover from cache) one response per prompt for an entrant.

    Cache misses are appended to the per-entrant JSONL immediately so a
    SIGINT / OOM mid-sweep loses at most the in-flight prompt; a re-run
    picks up where the file left off.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = _load_response_cache(cache_dir, spec.entrant_id)

    results: list[GeneratedResponse] = []
    fresh = 0
    n_tokens_total = 0
    n_resampled_total = 0
    t0 = time.perf_counter()

    for prompt in prompts:
        if prompt.prompt_id in cached:
            results.append(cached[prompt.prompt_id])
            continue

        # Vanilla-HF greedy is deterministic without a seed, but pinning
        # the global RNG keeps any auxiliary randomness reproducible per
        # prompt. PoE re-seeds internally via the seed kwarg.
        set_seed(prompt_seed(prompt.prompt_id))

        if spec.loader == "vanilla_hf":
            response_text, diagnostics = _generate_vanilla_hf(
                spec, loaded, prompt, max_new_tokens
            )
            p_correct: float | None = None
        else:
            response_text, p_correct, diagnostics = _generate_custom_poe(
                spec, loaded, prompt
            )

        record = GeneratedResponse(
            entrant_id=spec.entrant_id,
            prompt_id=prompt.prompt_id,
            response_text=response_text,
            p_correct=p_correct,
            diagnostics=diagnostics,
            timestamp=_now_iso(),
        )
        _append_response(cache_dir, spec.entrant_id, record)
        results.append(record)
        fresh += 1
        n_tokens_total += int(diagnostics.get("n_tokens_generated") or 0)
        n_resampled_total += int(diagnostics.get("n_resampled") or 0)

    elapsed = time.perf_counter() - t0
    _print_summary(
        spec.entrant_id,
        n_total=len(prompts),
        n_cached=len(prompts) - fresh,
        n_fresh=fresh,
        mean_tokens=(n_tokens_total / fresh) if fresh else 0.0,
        mean_resampled=(n_resampled_total / fresh) if fresh else 0.0,
        elapsed=elapsed,
    )
    return results


def _print_summary(
    entrant_id: str,
    *,
    n_total: int,
    n_cached: int,
    n_fresh: int,
    mean_tokens: float,
    mean_resampled: float,
    elapsed: float,
) -> None:
    print()
    print(f"=== {entrant_id} ===")
    print(f"{'total':<20} | {n_total}")
    print(f"{'cache hits':<20} | {n_cached}")
    print(f"{'fresh generations':<20} | {n_fresh}")
    print(f"{'mean tokens':<20} | {mean_tokens:.1f}")
    print(f"{'mean resampled':<20} | {mean_resampled:.2f}")
    print(f"{'wall time':<20} | {elapsed:.1f}s")


def _free_loaded(loaded: LoadedEntrant | None) -> None:
    """Drop references to a :class:`LoadedEntrant` and release GPU memory."""
    if loaded is None:
        return
    loaded.hf_model = None
    loaded.hydra = None
    loaded.poe = None
    gc.collect()
    torch.cuda.empty_cache()


def _group_specs_by_load(specs: list[EntrantSpec]) -> list[list[EntrantSpec]]:
    """Group entrants by ``(loader, rob_checkpoint)`` so shared models load once.

    Iteration order follows the input ordering of ``specs`` so the run
    log is predictable. Entrants 3 and 4 land in the same group:
    bypass_jury differs only at generation time, the underlying weights
    are identical.
    """
    groups: dict[tuple[str, int | None], list[EntrantSpec]] = {}
    order: list[tuple[str, int | None]] = []
    for spec in specs:
        key = (spec.loader, spec.rob_checkpoint)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(spec)
    return [groups[k] for k in order]


def run_generation(
    specs: list[EntrantSpec],
    prompts: list[Prompt],
    cache_dir: Path,
    max_new_tokens: int = 256,
) -> dict[str, list[GeneratedResponse]]:
    """Drive generation across all entrants with shared model loads.

    Entrants with the same ``(loader, rob_checkpoint)`` share one
    loaded model — only the eval-mode kwargs (``bypass_jury``,
    ``temperature``) differ between them. Loads happen group-by-group;
    GPU memory is explicitly released between groups so the peak
    footprint is one model at a time.
    """
    out: dict[str, list[GeneratedResponse]] = {}
    for group in _group_specs_by_load(specs):
        loaded: LoadedEntrant | None = None
        try:
            print(
                f"\nLoading model for group: "
                f"{[s.entrant_id for s in group]} "
                f"(loader={group[0].loader}, rob_checkpoint={group[0].rob_checkpoint})"
            )
            loaded = build_entrant(group[0], max_new_tokens=max_new_tokens)
            for spec in group:
                out[spec.entrant_id] = generate_responses_for_entrant(
                    spec, loaded, prompts, cache_dir, max_new_tokens=max_new_tokens
                )
        finally:
            _free_loaded(loaded)
            del loaded
    return out


def _parse_entrant_list(value: str) -> list[EntrantSpec]:
    ids = [s.strip() for s in value.split(",") if s.strip()]
    out: list[EntrantSpec] = []
    for eid in ids:
        if eid not in ENTRANTS_BY_ID:
            raise SystemExit(
                f"Unknown entrant id {eid!r}. Known: {sorted(ENTRANTS_BY_ID)}"
            )
        out.append(ENTRANTS_BY_ID[eid])
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank",
        type=Path,
        default=Path("olmo_tap/final_evals/elo/prompts/bank.jsonl"),
        help="Path to the prompt bank JSONL.",
    )
    parser.add_argument(
        "--entrants",
        type=str,
        default=",".join(s.entrant_id for s in ENTRANTS),
        help="Comma-separated entrant ids to run (default: all four).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("olmo_tap/final_evals/elo/caches/responses"),
        help="Directory for the per-entrant response JSONL cache.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Decoding budget per response.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only generate the first N prompts (smoke testing).",
    )
    args = parser.parse_args(argv)

    prompts = load_prompt_bank(args.bank)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    specs = _parse_entrant_list(args.entrants)
    print(
        f"Generating {len(specs)} entrant(s) x {len(prompts)} prompt(s) -> "
        f"{args.cache_dir}"
    )
    run_generation(specs, prompts, args.cache_dir, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()

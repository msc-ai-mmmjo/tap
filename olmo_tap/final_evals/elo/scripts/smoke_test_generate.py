"""Smoke test the four-entrant generation pipeline on a 3-prompt slice.

Confirms that all four entrants load and produce non-empty responses,
that the per-prompt seeding aligns the draft head across the three
Hydra entrants, that bypass_jury entrants record zero resampled
positions, that the full PoE entrant resamples on at least some
prompts, and that the vanilla-HF entrant is deterministic across
identical-seed runs.

The cache directory is forced to a smoke-test-only location so the
real response cache is untouched. Re-runs of the smoke test reuse
that scratch cache; delete it to force fresh generation.

Run::

    pixi run -e cuda python -m olmo_tap.final_evals.elo.scripts.smoke_test_generate
"""

from __future__ import annotations

import shutil
from pathlib import Path

from olmo_tap.final_evals.elo.entrants import ENTRANTS, get_entrant
from olmo_tap.final_evals.elo.generate import (
    GeneratedResponse,
    Prompt,
    load_prompt_bank,
    run_generation,
)


SMOKE_CACHE_DIR = Path("olmo_tap/final_evals/elo/caches/smoke_responses")
SMOKE_CACHE_DIR_RERUN = Path("olmo_tap/final_evals/elo/caches/smoke_responses_rerun")


def _select_smoke_prompts(bank: list[Prompt]) -> list[Prompt]:
    """Pick one prompt from each source so the smoke covers the full bank shape.

    Falls back to the first three prompts if any source is missing.
    """
    by_source: dict[str, Prompt] = {}
    for p in bank:
        by_source.setdefault(p.source, p)
    chosen: list[Prompt] = []
    for src in ("medmcqa_open", "medqa", "curated"):
        if src in by_source:
            chosen.append(by_source[src])
    if len(chosen) < 3:
        # Pad with leading prompts not already in chosen.
        seen = {p.prompt_id for p in chosen}
        for p in bank:
            if p.prompt_id in seen:
                continue
            chosen.append(p)
            if len(chosen) == 3:
                break
    return chosen[:3]


def _check_seed_alignment(
    results: dict[str, list[GeneratedResponse]],
) -> list[str]:
    """Assert entrants 2/3/4 selected the same draft head per prompt."""
    issues: list[str] = []
    hydra_ids = ["security_only", "security_plus_robustness", "full_poe"]
    if not all(eid in results for eid in hydra_ids):
        return issues  # subset run, skip the check
    by_prompt: dict[str, dict[str, int | None]] = {}
    for eid in hydra_ids:
        for rec in results[eid]:
            heads = by_prompt.setdefault(rec.prompt_id, {})
            heads[eid] = rec.diagnostics.get("draft_head_idx")
    for prompt_id, heads in by_prompt.items():
        unique = set(heads.values())
        if len(unique) > 1:
            issues.append(f"draft head mismatch on {prompt_id}: {heads}")
    return issues


def _check_bypass_resampling(
    results: dict[str, list[GeneratedResponse]],
) -> list[str]:
    """Bypass-jury entrants must record zero resampled positions."""
    issues: list[str] = []
    for eid in ("security_only", "security_plus_robustness"):
        if eid not in results:
            continue
        for rec in results[eid]:
            n = rec.diagnostics.get("n_resampled", 0)
            if n != 0:
                issues.append(
                    f"{eid} should bypass the jury but n_resampled={n} on {rec.prompt_id}"
                )
    return issues


def _check_full_poe_resamples(
    results: dict[str, list[GeneratedResponse]],
) -> list[str]:
    """Full PoE should resample on at least one of the three smoke prompts."""
    if "full_poe" not in results:
        return []
    total = sum(rec.diagnostics.get("n_resampled", 0) for rec in results["full_poe"])
    if total == 0:
        return [
            "full_poe recorded n_resampled=0 across all smoke prompts -- "
            "expected the jury to reject at least once"
        ]
    return []


def _check_nonempty(
    results: dict[str, list[GeneratedResponse]],
) -> list[str]:
    issues: list[str] = []
    for eid, recs in results.items():
        for rec in recs:
            if not rec.response_text.strip():
                issues.append(f"{eid} produced an empty response on {rec.prompt_id}")
    return issues


def _check_vanilla_determinism(
    first: dict[str, list[GeneratedResponse]],
    second: dict[str, list[GeneratedResponse]],
) -> list[str]:
    issues: list[str] = []
    if "base_olmo" not in first or "base_olmo" not in second:
        return issues
    by_id_a = {r.prompt_id: r.response_text for r in first["base_olmo"]}
    by_id_b = {r.prompt_id: r.response_text for r in second["base_olmo"]}
    for pid, text_a in by_id_a.items():
        text_b = by_id_b.get(pid)
        if text_b != text_a:
            issues.append(
                f"vanilla_hf nondeterministic on {pid}: {text_a!r} != {text_b!r}"
            )
    return issues


def _print_response_preview(results: dict[str, list[GeneratedResponse]]) -> None:
    print("\n--- Response preview ---")
    for eid, recs in results.items():
        print(f"\n[{eid}]")
        for rec in recs:
            preview = rec.response_text.strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:197] + "..."
            print(f"  {rec.prompt_id}: {preview}")
            print(f"    diag: {rec.diagnostics}")


def main() -> None:
    bank_path = Path("olmo_tap/final_evals/elo/prompts/bank.jsonl")
    bank = load_prompt_bank(bank_path)
    prompts = _select_smoke_prompts(bank)

    print(f"Smoke prompts ({len(prompts)}):")
    for p in prompts:
        print(f"  - {p.prompt_id} [{p.source}] {p.text[:80]}")

    specs = list(ENTRANTS)

    if SMOKE_CACHE_DIR.exists():
        shutil.rmtree(SMOKE_CACHE_DIR)
    if SMOKE_CACHE_DIR_RERUN.exists():
        shutil.rmtree(SMOKE_CACHE_DIR_RERUN)

    print("\n=== First pass ===")
    first = run_generation(specs, prompts, SMOKE_CACHE_DIR, max_new_tokens=128)

    # Re-run only the vanilla entrant in a fresh cache dir to confirm
    # determinism. Using a separate dir avoids cache hits short-circuiting
    # the regeneration. Keeping it limited to vanilla because re-loading
    # OLMo to re-test full PoE on the same fixed seed would just confirm
    # what the seed-determinism unit test already covers.
    print("\n=== Second pass (vanilla_hf only, for determinism) ===")
    second = run_generation(
        [get_entrant("base_olmo")], prompts, SMOKE_CACHE_DIR_RERUN, max_new_tokens=128
    )

    issues: list[str] = []
    issues += _check_nonempty(first)
    issues += _check_seed_alignment(first)
    issues += _check_bypass_resampling(first)
    issues += _check_full_poe_resamples(first)
    issues += _check_vanilla_determinism(first, second)

    _print_response_preview(first)

    print("\n=== Smoke test results ===")
    if issues:
        print(f"FAILED -- {len(issues)} issue(s):")
        for msg in issues:
            print(f"  ! {msg}")
        raise SystemExit(1)
    print("OK -- all checks passed")


if __name__ == "__main__":
    main()

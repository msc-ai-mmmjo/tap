"""Tournament orchestrator: response cache → judges → Elo → artifacts.

Wires together the existing per-entrant response cache, the
LLM-judge batch pipeline, and the local permutation-averaged Elo
engine. Two run modes are supported:

  * ``pilot``: 50-prompt stratified subset judged by Sonnet, no
    prompt-cache pre-warming. Used as a cheap sanity gate before the
    headline run.
  * ``headline``: the full 143-prompt bank judged by Opus, with a
    1-query pre-warm batch per rubric so Anthropic's prompt cache is
    populated before the bulk batch goes out.

Output directory layout::

    runs/<mode>_<timestamp>/
    ├── manifest.json         # mode, model, prompt count, seeds, timestamps
    ├── matches/matches.jsonl # one record per (dimension, pair)
    ├── verdicts/{factuality,calibration,clinical_utility}.jsonl
    ├── elo_results.json      # mean ± SEM per entrant per dimension
    ├── elo_per_perm.npz      # full (n_perms × entrants) traces
    └── pairwise_winrates.csv # per-pair win/loss/tie counts

Usage::

    pixi run -e default python -m olmo_tap.final_evals.elo.run_tournament \\
        --config olmo_tap/final_evals/elo/configs/tournament1.yaml \\
        --mode pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import yaml

from olmo_tap.final_evals.elo.elo_engine import (
    DEFAULT_INITIAL_RATING,
    DEFAULT_K_SWEEP,
    EloResult,
    Match,
    compute_elo_permutation,
    k_factor_sweep,
)
from olmo_tap.final_evals.elo.judge import (
    DIMENSIONS,
    Dimension,
    JudgeConfig,
    Judgment,
    PairToJudge,
    Rubric,
    _judgment_as_dict,
    judge_pairs,
)
from olmo_tap.final_evals.elo.match_builder import (
    DEFAULT_PILOT_SEED,
    build_match_list,
    select_pilot_subset,
)
from olmo_tap.final_evals.elo.types import (
    GeneratedResponse,
    load_prompt_bank,
)


logger = logging.getLogger(__name__)

Mode = Literal["pilot", "headline"]
MODES: tuple[Mode, ...] = ("pilot", "headline")


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a configuration-level Elo tournament against the cached "
            "responses produced by the generate pipeline."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a tournament YAML config (see configs/tournament1.yaml).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=MODES,
        required=True,
        help=(
            "`pilot` runs Sonnet against the 50-prompt subset; `headline` "
            "runs Opus against the full bank with rubric-cache pre-warming."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for run artifacts. Defaults to "
            "`olmo_tap/final_evals/elo/runs/<mode>_<timestamp>`."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override the judge model id selected by --mode.",
    )
    parser.add_argument(
        "--n-perms",
        type=int,
        default=None,
        help="Override `elo.n_perms` from the config.",
    )
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help=(
            "Cap the per-dimension pair list at the first N entries. "
            "Diagnostic aid only — leave unset for normal runs."
        ),
    )
    parser.add_argument(
        "--skip-judging",
        action="store_true",
        help=(
            "Skip the judge step entirely; useful for re-running the Elo "
            "math from already-populated verdict caches."
        ),
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Config + cache loading
# --------------------------------------------------------------------------- #


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_response_cache(
    cache_dir: Path, entrants: Iterable[str]
) -> dict[tuple[str, str], GeneratedResponse]:
    """Load every per-entrant ``responses_<entrant>.jsonl`` into a flat dict.

    Missing per-entrant files are reported and skipped; downstream
    :func:`match_builder.build_match_list` decides whether to abort by
    counting how many ``(entrant, prompt)`` cells are present.
    """
    cache: dict[tuple[str, str], GeneratedResponse] = {}
    for eid in entrants:
        path = cache_dir / f"responses_{eid}.jsonl"
        if not path.exists():
            logger.error("Missing response cache for entrant %s at %s", eid, path)
            continue
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                cache[(eid, record["prompt_id"])] = GeneratedResponse(**record)
    return cache


# --------------------------------------------------------------------------- #
# On-disk artifact writers
# --------------------------------------------------------------------------- #


def write_manifest(
    out_dir: Path,
    *,
    mode: Mode,
    config_path: Path,
    judge_model: str,
    entrants: list[str],
    bank_size: int,
    selected_size: int,
    n_perms: int,
    seed: int,
    pilot_seed: int,
    started_at: str,
    finished_at: str | None,
) -> None:
    payload = {
        "mode": mode,
        "config_path": str(config_path),
        "judge_model": judge_model,
        "entrants": entrants,
        "bank_size": bank_size,
        "selected_prompt_count": selected_size,
        "n_perms": n_perms,
        "shuffle_seed": seed,
        "pilot_seed": pilot_seed,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_matches(
    out_dir: Path, matches_by_dim: dict[Dimension, list[PairToJudge]]
) -> None:
    matches_dir = out_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)
    path = matches_dir / "matches.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for dimension, pairs in matches_by_dim.items():
            for pair in pairs:
                row = {"dimension": dimension, **asdict(pair)}
                fh.write(json.dumps(row, ensure_ascii=False))
                fh.write("\n")


def write_verdicts(
    out_dir: Path, dimension: Dimension, judgments: list[Judgment]
) -> None:
    verdicts_dir = out_dir / "verdicts"
    verdicts_dir.mkdir(parents=True, exist_ok=True)
    path = verdicts_dir / f"{dimension}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for j in judgments:
            fh.write(json.dumps(_judgment_as_dict(j), ensure_ascii=False))
            fh.write("\n")


# --------------------------------------------------------------------------- #
# Reconciliation: Judgment → Elo Match
# --------------------------------------------------------------------------- #


def judgments_to_matches(judgments: Iterable[Judgment]) -> list[Match]:
    """Drop ties / inconsistent verdicts; return ``(a, b, winner)`` triples."""
    out: list[Match] = []
    for j in judgments:
        if j.winner is None:
            continue
        out.append((j.entrant_a, j.entrant_b, j.winner))
    return out


def pairwise_winrates(
    judgments: Iterable[Judgment],
) -> dict[tuple[str, str], dict[str, int]]:
    """Aggregate per-pair counts. ``(a, b)`` keys are sorted lexicographically
    so ``(A, B)`` and ``(B, A)`` collapse onto one row.
    """
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"a_wins": 0, "b_wins": 0, "ties": 0, "inconsistent": 0}
    )
    for j in judgments:
        a, b = sorted((j.entrant_a, j.entrant_b))
        cell = counts[(a, b)]
        if j.inconsistent:
            cell["inconsistent"] += 1
            cell["ties"] += 1
            continue
        if j.winner is None:
            cell["ties"] += 1
        elif j.winner == a:
            cell["a_wins"] += 1
        else:
            cell["b_wins"] += 1
    return counts


def write_pairwise_winrates_csv(
    out_dir: Path,
    by_dim: dict[Dimension, dict[tuple[str, str], dict[str, int]]],
) -> None:
    path = out_dir / "pairwise_winrates.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "dimension",
                "entrant_a",
                "entrant_b",
                "a_wins",
                "b_wins",
                "ties",
                "inconsistent",
            ]
        )
        for dimension, pairs in by_dim.items():
            for (a, b), cell in sorted(pairs.items()):
                writer.writerow(
                    [
                        dimension,
                        a,
                        b,
                        cell["a_wins"],
                        cell["b_wins"],
                        cell["ties"],
                        cell["inconsistent"],
                    ]
                )


# --------------------------------------------------------------------------- #
# Elo result serialisation
# --------------------------------------------------------------------------- #


def _elo_result_to_dict(result: EloResult) -> dict[str, float | str]:
    return {
        "entrant_id": result.entrant_id,
        "mean": result.mean,
        "sem": result.sem,
        "ci95_low": result.ci95_low,
        "ci95_high": result.ci95_high,
    }


def write_elo_results(
    out_dir: Path,
    *,
    primary: dict[Dimension, dict[str, EloResult]],
    sweep: dict[Dimension, dict[float, dict[str, EloResult]]],
    n_matches: dict[Dimension, int],
    default_k: float,
) -> None:
    payload: dict[str, Any] = {
        "default_k": default_k,
        "primary": {
            dim: {eid: _elo_result_to_dict(r) for eid, r in res.items()}
            for dim, res in primary.items()
        },
        "k_sensitivity": {
            dim: {
                str(k): {eid: _elo_result_to_dict(r) for eid, r in inner.items()}
                for k, inner in by_k.items()
            }
            for dim, by_k in sweep.items()
        },
        "n_decisive_matches": dict(n_matches),
    }
    (out_dir / "elo_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_elo_per_perm(
    out_dir: Path, primary: dict[Dimension, dict[str, EloResult]]
) -> None:
    """Persist the full ``(n_perms × entrants)`` rating array per dimension.

    Saved as a single ``.npz`` archive with two keys per dimension:
    ``<dim>__data`` (float64, shape ``(n_perms, n_entrants)``) and
    ``<dim>__entrants`` (string array of column ids).
    """
    arrays: dict[str, np.ndarray] = {}
    for dimension, results in primary.items():
        if not results:
            continue
        eids = sorted(results.keys())
        stacked = np.stack([results[eid].per_perm_ratings for eid in eids], axis=1)
        arrays[f"{dimension}__data"] = stacked
        arrays[f"{dimension}__entrants"] = np.array(eids)
    if arrays:
        np.savez(out_dir / "elo_per_perm.npz", **arrays)


# --------------------------------------------------------------------------- #
# Stdout summary
# --------------------------------------------------------------------------- #


def print_summary_table(
    primary: dict[Dimension, dict[str, EloResult]],
    n_matches: dict[Dimension, int],
) -> None:
    print()
    print("=" * 72)
    print("Elo summary (mean ± SEM)")
    print("=" * 72)
    if not primary or not any(primary.values()):
        print("(no Elo results)")
        return
    all_entrants: set[str] = set()
    for results in primary.values():
        all_entrants.update(results.keys())
    entrants = sorted(all_entrants)
    dims = list(primary.keys())

    header = f"{'entrant':<28}" + "".join(f" | {dim:>22}" for dim in dims)
    print(header)
    print("-" * len(header))
    for eid in entrants:
        row = f"{eid:<28}"
        for dim in dims:
            r = primary.get(dim, {}).get(eid)
            cell = "—" if r is None else f"{r.mean:8.1f} ± {r.sem:5.2f}"
            row += f" | {cell:>22}"
        print(row)
    print("-" * len(header))
    counts = "decisive matches: " + ", ".join(
        f"{dim}={n_matches.get(dim, 0)}" for dim in dims
    )
    print(counts)


# --------------------------------------------------------------------------- #
# Pre-warming (headline mode only)
# --------------------------------------------------------------------------- #


def _prewarm_rubric_cache(
    pairs: list[PairToJudge],
    rubric: Rubric,
    config: JudgeConfig,
) -> None:
    """Submit a tiny pre-warm batch so Anthropic's prompt cache is populated
    with the rubric prefix before the bulk batch goes out.

    Used only in headline mode — the pilot is fast enough that the wait on
    a pre-warm batch outweighs the cache hit savings.
    """
    if not pairs:
        return
    logger.info("Pre-warming prompt cache for %s with 1-pair batch", rubric.dimension)
    judge_pairs(pairs[:1], rubric, config)


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config = load_config(args.config)
    mode: Mode = args.mode

    judge_cfg = config["judge"]
    elo_cfg = config["elo"]
    rubrics_cfg = config["rubrics"]
    caches_cfg = config["caches"]

    judge_model = args.judge_model or (
        judge_cfg["pilot_model"] if mode == "pilot" else judge_cfg["headline_model"]
    )
    n_perms = args.n_perms if args.n_perms is not None else int(elo_cfg["n_perms"])
    shuffle_seed = int(elo_cfg.get("shuffle_seed", 0))
    initial_rating = float(elo_cfg.get("initial_rating", DEFAULT_INITIAL_RATING))
    default_k = float(elo_cfg.get("default_k", 16))
    k_sweep = tuple(elo_cfg.get("k_factor_sweep", DEFAULT_K_SWEEP))

    runs_default = Path("olmo_tap/final_evals/elo/runs")
    if args.output_dir is None:
        out_dir = runs_default / f"{mode}_{_now_stamp()}"
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matches").mkdir(parents=True, exist_ok=True)
    (out_dir / "verdicts").mkdir(parents=True, exist_ok=True)

    started_at = _now_iso()
    logger.info("Run start: mode=%s out=%s judge=%s", mode, out_dir, judge_model)

    bank_path = Path(config["prompts"]["bank_path"])
    full_bank = load_prompt_bank(bank_path)
    bank_size = len(full_bank)
    entrants = [e["entrant_id"] for e in config["entrants"]]
    response_cache = load_response_cache(Path(caches_cfg["responses_dir"]), entrants)

    expected = len(entrants) * len(full_bank)
    if len(response_cache) < expected:
        logger.warning(
            "Response cache covers %d/%d (entrant, prompt) cells. "
            "Missing rows will be dropped from the match list.",
            len(response_cache),
            expected,
        )

    if mode == "pilot":
        bank = select_pilot_subset(full_bank)
        logger.info("Pilot subset: %d prompts", len(bank))
    else:
        bank = full_bank

    matches_by_dim = build_match_list(
        bank, response_cache, entrants, dimensions=DIMENSIONS
    )
    if args.limit_pairs is not None:
        matches_by_dim = {
            d: pairs[: args.limit_pairs] for d, pairs in matches_by_dim.items()
        }
    write_matches(out_dir, matches_by_dim)
    for dim, pairs in matches_by_dim.items():
        logger.info(
            "Dimension %s: %d unordered pairs (≈ %d judge queries)",
            dim,
            len(pairs),
            2 * len(pairs),
        )

    # Initial manifest — finished_at gets filled in at the end.
    write_manifest(
        out_dir,
        mode=mode,
        config_path=args.config,
        judge_model=judge_model,
        entrants=entrants,
        bank_size=bank_size,
        selected_size=len(bank),
        n_perms=n_perms,
        seed=shuffle_seed,
        pilot_seed=DEFAULT_PILOT_SEED,
        started_at=started_at,
        finished_at=None,
    )

    # ---- Judging ----
    judge_cache_dir = Path(caches_cfg["judgments_dir"])
    judge_cache_dir.mkdir(parents=True, exist_ok=True)
    j_config = JudgeConfig(judge_model=judge_model, cache_dir=judge_cache_dir)

    judgments_by_dim: dict[Dimension, list[Judgment]] = {}
    if args.skip_judging:
        logger.warning("--skip-judging set; no judging performed this run.")
        for dimension in DIMENSIONS:
            judgments_by_dim[dimension] = []
    else:
        for dimension in DIMENSIONS:
            pairs = matches_by_dim.get(dimension, [])
            if not pairs:
                judgments_by_dim[dimension] = []
                continue
            rubric = Rubric.load(dimension, Path(rubrics_cfg[dimension]["path"]))
            if mode == "headline":
                _prewarm_rubric_cache(pairs, rubric, j_config)
            logger.info(
                "Judging %d pairs on %s with %s", len(pairs), dimension, judge_model
            )
            result = judge_pairs(pairs, rubric, j_config)
            judgments_by_dim[dimension] = result.judgments
            write_verdicts(out_dir, dimension, result.judgments)
            stats = result.cache_stats
            logger.info(
                "Dimension %s done: cache_hits=%d fresh=%d "
                "cache_creation=%d cache_read=%d input=%d output=%d",
                dimension,
                stats.cache_hits,
                stats.fresh_calls,
                stats.cache_creation_input_tokens,
                stats.cache_read_input_tokens,
                stats.input_tokens,
                stats.output_tokens,
            )

    # ---- Elo ----
    primary: dict[Dimension, dict[str, EloResult]] = {}
    sweep: dict[Dimension, dict[float, dict[str, EloResult]]] = {}
    n_matches: dict[Dimension, int] = {}
    pairwise_by_dim: dict[Dimension, dict[tuple[str, str], dict[str, int]]] = {}

    for dimension in DIMENSIONS:
        judgments = judgments_by_dim.get(dimension, [])
        elo_matches = judgments_to_matches(judgments)
        n_matches[dimension] = len(elo_matches)
        pairwise_by_dim[dimension] = pairwise_winrates(judgments)
        if not elo_matches:
            logger.warning(
                "Dimension %s: no decisive matches; skipping Elo.", dimension
            )
            primary[dimension] = {}
            sweep[dimension] = {}
            continue
        primary[dimension] = compute_elo_permutation(
            elo_matches,
            k=default_k,
            initial_rating=initial_rating,
            n_perms=n_perms,
            seed=shuffle_seed,
        )
        sweep[dimension] = k_factor_sweep(
            elo_matches,
            k_values=tuple(int(k) for k in k_sweep),
            initial_rating=initial_rating,
            n_perms=n_perms,
            seed=shuffle_seed,
        )

    write_elo_results(
        out_dir,
        primary=primary,
        sweep=sweep,
        n_matches=n_matches,
        default_k=default_k,
    )
    write_elo_per_perm(out_dir, primary)
    write_pairwise_winrates_csv(out_dir, pairwise_by_dim)

    print_summary_table(primary, n_matches)

    write_manifest(
        out_dir,
        mode=mode,
        config_path=args.config,
        judge_model=judge_model,
        entrants=entrants,
        bank_size=bank_size,
        selected_size=len(bank),
        n_perms=n_perms,
        seed=shuffle_seed,
        pilot_seed=DEFAULT_PILOT_SEED,
        started_at=started_at,
        finished_at=_now_iso(),
    )
    logger.info("Run complete: %s", out_dir)


if __name__ == "__main__":
    main()

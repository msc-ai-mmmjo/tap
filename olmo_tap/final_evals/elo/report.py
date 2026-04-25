"""Tournament reporting helpers.

Stubbed for now. Will produce:

  - ``elo_results.json`` — mean / SEM / 95% CI per entrant per dimension,
    plus the full per-permutation traces.
  - ``sensitivity_heatmap.png`` — K × entrant heatmap mirroring the
    ranking-stability figure from Boubdir et al. (2023).
  - ``pairwise_winrates.csv`` — raw win/loss/tie counts per pair per
    dimension (pre-Elo).
  - ``judge_log.jsonl`` — every judge query with full inputs, verdict,
    reasoning trace, and cache key.
  - ``run_manifest.json`` — timestamps, seeds, model versions, prompt-set
    hash, rubric version (so reviewers can verify reproducibility).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from olmo_tap.final_evals.elo.elo_engine import EloResult


def write_results_json(
    results_per_dim: Mapping[str, dict[str, EloResult]],
    out_path: Path,
) -> None:
    """Serialise per-dimension Elo results to ``out_path``."""
    raise NotImplementedError(
        "write_results_json is not yet implemented — pending the reporting build-out."
    )


def render_sensitivity_heatmap(
    sweep: Mapping[float, dict[str, EloResult]],
    out_path: Path,
    *,
    dimension: str,
) -> None:
    """Render the K × entrant heatmap (one PNG per dimension)."""
    raise NotImplementedError(
        "render_sensitivity_heatmap is not yet implemented — pending the "
        "reporting build-out."
    )


def write_run_manifest(
    config: Mapping[str, Any],
    out_path: Path,
) -> None:
    """Snapshot every reproducibility-relevant input for the report."""
    raise NotImplementedError(
        "write_run_manifest is not yet implemented — pending the reporting build-out."
    )

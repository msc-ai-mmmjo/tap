"""Tournament orchestrator: generate -> judge -> elo -> report.

Wires together the response generation pipeline, the Anthropic-judge
pipeline, and the local Elo engine. Stubs ``main()`` so the pixi task is
registered and the CLI signature is fixed; the orchestration logic lands
once :mod:`generate` and :mod:`judge` ship.

Usage::

    python -m olmo_tap.final_evals.elo.run_tournament \\
        --config olmo_tap/final_evals/elo/configs/tournament1.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a tournament YAML config (see configs/tournament1.yaml).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=(
            "Override the judge model id (e.g. 'claude-sonnet-4-6' for the "
            "pilot, 'claude-opus-4-7' for the headline run)."
        ),
    )
    parser.add_argument(
        "--n-perms",
        type=int,
        default=None,
        help="Override the number of Elo permutations from the config.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("olmo_tap/final_evals/elo/runs"),
        help="Where to write run artifacts (results.json, heatmap, manifest).",
    )
    return parser.parse_args()


def main() -> None:
    """Drive the full tournament pipeline."""
    args = parse_args()  # noqa: F841 — args consumed once orchestration lands
    raise NotImplementedError(
        "run_tournament.main is not yet implemented — pending the generate "
        "and judge pipelines."
    )


if __name__ == "__main__":
    main()

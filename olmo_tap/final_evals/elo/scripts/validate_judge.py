"""End-to-end validation of the judge pipeline against a hand-crafted pair.

Builds one ``(response_a, response_b)`` pair where response_a is clearly
better on every dimension (a coherent medical answer vs. a nonsense
reply), runs ``judge_pairs`` against all three rubrics with
Sonnet 4.6, and asserts that the verdict is ``A`` for each. Reasoning
traces and cache stats are printed so a human can eyeball the calls.

Run with::

    pixi run -e default python -m olmo_tap.final_evals.elo.scripts.validate_judge

The script must use Sonnet (cheap); Opus is reserved for the headline
run of the tournament. Each invocation costs about $0.01.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from olmo_tap.final_evals.elo.judge import (
    DIMENSIONS,
    Dimension,
    JudgeConfig,
    PairToJudge,
    Rubric,
    judge_pairs,
)

REPO_ROOT = Path(__file__).resolve().parents[4]

RUBRIC_PATHS: dict[Dimension, Path] = {
    "factuality": REPO_ROOT
    / "olmo_tap/final_evals/elo/configs/rubrics/factuality.txt",
    "calibration": REPO_ROOT
    / "olmo_tap/final_evals/elo/configs/rubrics/calibration.txt",
    "clinical_utility": REPO_ROOT
    / "olmo_tap/final_evals/elo/configs/rubrics/clinical_utility.txt",
}

DEFAULT_CACHE_DIR = REPO_ROOT / "olmo_tap/final_evals/elo/caches/judgments_validation"


GOOD_RESPONSE = (
    "The first-line treatment for primary hypothyroidism is levothyroxine "
    "(synthetic T4), taken once daily on an empty stomach. Typical starting "
    "doses are 1.6 mcg/kg/day in healthy adults, with lower starting doses "
    "(25–50 mcg/day) in older adults or patients with cardiovascular disease "
    "to avoid precipitating angina or arrhythmia. Dose adjustments are "
    "guided by TSH measured 6–8 weeks after each change. Levothyroxine is "
    "preferred over T3 because of its long half-life and smoother "
    "physiologic profile. If the patient remains symptomatic despite a "
    "TSH in target range, evaluate for adherence, malabsorption, or "
    "interfering medications before considering combination T4/T3 therapy."
)

BAD_RESPONSE = (
    "Hypothyroidism is when the thyroid is sleepy. Banana smoothies and "
    "sunlight are the primary treatment. If the symptoms persist for more "
    "than a fortnight, ask the patient to whisper their TSH value into a "
    "glass of warm water and discard. Aspirin may also help. The thyroid "
    "is located in the abdomen and produces insulin."
)


def _build_pair() -> PairToJudge:
    return PairToJudge(
        prompt_id="validation_hypothyroidism",
        source="medqa",  # non-curated so all three rubrics process the pair
        prompt_text=(
            "What is the first-line pharmacologic treatment for primary "
            "hypothyroidism in an otherwise healthy adult, and how is it "
            "dosed and monitored?"
        ),
        entrant_a="coherent_response",
        entrant_b="nonsense_response",
        response_a=GOOD_RESPONSE,
        response_b=BAD_RESPONSE,
        gold_answer=(
            "Levothyroxine (synthetic T4), once-daily oral; titrated to a "
            "TSH within the laboratory reference range, with TSH rechecked "
            "6–8 weeks after dose changes."
        ),
    )


def _print_separator() -> None:
    print("-" * 78)


def _summarise(result, dimension: Dimension) -> bool:
    if not result.judgments:
        print(f"[{dimension}] No judgments returned (rubric likely filtered).")
        return False
    judgment = result.judgments[0]
    raw_forward, raw_swapped = judgment.raw
    print(f"[{dimension}] winner = {judgment.winner!r}, inconsistent = {judgment.inconsistent}")
    print(
        f"[{dimension}] forward verdict = {raw_forward.verdict}, "
        f"swapped verdict = {raw_swapped.verdict}"
    )
    print(f"[{dimension}] cache_stats = {result.cache_stats}")
    print(f"[{dimension}] forward reasoning (truncated):")
    print(raw_forward.reasoning[:600])
    if len(raw_forward.reasoning) > 600:
        print("... (truncated)")
    print(f"[{dimension}] swapped reasoning (truncated):")
    print(raw_swapped.reasoning[:600])
    if len(raw_swapped.reasoning) > 600:
        print("... (truncated)")
    return judgment.winner == "coherent_response"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge-model",
        default="claude-sonnet-4-6",
        help=(
            "Anthropic model id for the judge. Validation should use Sonnet; "
            "Opus is reserved for the headline tournament run."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory to write the validation judgment cache JSONLs.",
    )
    args = parser.parse_args()

    if "opus" in args.judge_model.lower():
        print(
            "Refusing to run validation with an Opus model. Use Sonnet for "
            "validation; Opus is reserved for the headline run.",
            file=sys.stderr,
        )
        return 2

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    pair = _build_pair()
    config = JudgeConfig(
        judge_model=args.judge_model,
        cache_dir=args.cache_dir,
    )

    all_passed = True
    total_stats = {
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "fresh_calls": 0,
        "cache_hits": 0,
    }

    for dimension in DIMENSIONS:
        _print_separator()
        rubric_path = RUBRIC_PATHS[dimension]
        rubric = Rubric.load(dimension, rubric_path)
        result = judge_pairs(pairs=[pair], rubric=rubric, config=config)
        passed = _summarise(result, dimension)
        all_passed = all_passed and passed
        total_stats["cache_creation_input_tokens"] += result.cache_stats.cache_creation_input_tokens
        total_stats["cache_read_input_tokens"] += result.cache_stats.cache_read_input_tokens
        total_stats["input_tokens"] += result.cache_stats.input_tokens
        total_stats["output_tokens"] += result.cache_stats.output_tokens
        total_stats["fresh_calls"] += result.cache_stats.fresh_calls
        total_stats["cache_hits"] += result.cache_stats.cache_hits

    _print_separator()
    print("Aggregate cache stats:")
    for key, value in total_stats.items():
        print(f"  {key}: {value}")

    if all_passed:
        print("\nAll three rubrics returned the expected winner ('coherent_response').")
        return 0
    print("\nAt least one rubric did not return the expected winner.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

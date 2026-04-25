"""Materialise the Tournament 1 prompt bank.

Source A — 80 MedMCQA validation items recast as open-ended clinical
questions (stratified-random by ``subject_name``).
Source B — 70 MedQA-USMLE vignettes from the test split, used in their
native open-ended form.
Source C — Hand-curated prompts, appended later by separate tooling.
This script is deterministic given ``--seed``; it must be re-runnable to
regenerate ``bank.jsonl`` byte-for-byte.

The output is a JSON-Lines file at ``--output`` (default
``olmo_tap/final_evals/elo/prompts/bank.jsonl``)::

    {
      "prompt_id":         "srcA_00001",
      "source":            "medmcqa_open" | "medqa" | "curated",
      "subject":           "endocrinology" | ...,
      "text":              "...",
      "gold_answer":       "...",         # Sources A and B
      "expected_behavior": "...",         # Source C only
      "generation_method": "...",         # Source C only
      "tags":              ["..."]
    }

This script produces only Sources A and B (150 prompts).  The curated
Source C is appended later by separate tooling; the loader downstream is
expected to ``json.loads`` each line of ``bank.jsonl`` and merge any
additional ``source: curated`` entries seamlessly.

Usage::

    python -m olmo_tap.final_evals.elo.prompts.build_prompt_set \\
        --output olmo_tap/final_evals/elo/prompts/bank.jsonl \\
        --seed 20260425
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset  # type: ignore[import-not-found]

from olmo_tap.constants import MCQ_LETTERS

DEFAULT_SEED: int = 20260425
DEFAULT_OUTPUT: Path = Path(__file__).resolve().parent / "bank.jsonl"
SOURCE_A_SIZE: int = 80
SOURCE_B_SIZE: int = 70
MEDMCQA_REPO: str = "openlifescienceai/medmcqa"
MEDMCQA_SPLIT: str = "validation"
MEDQA_REPO: str = "GBaker/MedQA-USMLE-4-options"
MEDQA_SPLIT: str = "test"

# Heuristic substitutions for MCQ → open-ended recasting.  Applied in order;
# only the first matching pattern fires per stem.  We deliberately keep this
# minimal — the bulk of the open-endedness comes from dropping the lettered
# options, not from rewording the stem.
_RECAST_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^which\s+of\s+the\s+following\s+is\s+", re.IGNORECASE), "What is "),
    (
        re.compile(r"^which\s+of\s+the\s+following\s+are\s+", re.IGNORECASE),
        "What are ",
    ),
    (
        re.compile(
            r"^which\s+of\s+the\s+following\s+(?:would|could|should|can|may|might)\s+",
            re.IGNORECASE,
        ),
        "What ",
    ),
    (
        re.compile(r"^which\s+of\s+the\s+following\s+", re.IGNORECASE),
        "Which ",
    ),
    (
        re.compile(r"^all\s+of\s+the\s+following\s+(?:are|is)\s+", re.IGNORECASE),
        "All of these are ",
    ),
)


def _strip_options_tail(stem: str) -> str:
    """Drop trailing ``A) ... B) ...`` option blocks if they slipped in."""
    # MedMCQA stems are usually clean but defensive parsing here costs little.
    pattern = re.compile(
        r"\s*(?:[\(\[]?\s*[A-D]\s*[\)\].:-]\s+).+",
        flags=re.DOTALL,
    )
    return pattern.split(stem, maxsplit=1)[0].strip()


def _recast_stem(stem: str) -> str:
    """Convert a multiple-choice stem into an open-ended clinical question."""
    cleaned = _strip_options_tail(stem.strip())
    # Strip a trailing colon that sometimes precedes the option list.
    cleaned = cleaned.rstrip(":").strip()
    for pattern, repl in _RECAST_PATTERNS:
        replaced, count = pattern.subn(repl, cleaned, count=1)
        if count:
            cleaned = replaced
            break
    if not cleaned.endswith(("?", ".")):
        cleaned += "?"
    # Append an open-ended prompt to discourage one-letter answers.
    return f"{cleaned} Briefly explain the reasoning."


def _stratified_sample(
    rows: list[dict[str, Any]],
    key: str,
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Stratified-random sample of ``n`` rows by ``rows[i][key]``.

    Falls back gracefully when bucket sizes round to zero or when the
    requested ``n`` exceeds the available rows: the function tops up the
    sample uniformly at random from the remaining rows.
    """
    if n > len(rows):
        raise ValueError(f"Cannot sample {n} rows from a population of {len(rows)}.")

    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[str(r.get(key) or "_unknown")].append(r)

    total = len(rows)
    bucket_keys = sorted(buckets)

    # Proportional allocation with explicit rounding control so we always end
    # up with exactly ``n`` items.
    targets: dict[str, int] = {}
    running = 0
    for k in bucket_keys[:-1]:
        share = round(n * len(buckets[k]) / total)
        share = min(share, len(buckets[k]))
        targets[k] = share
        running += share
    last_key = bucket_keys[-1]
    targets[last_key] = min(max(n - running, 0), len(buckets[last_key]))

    sampled: list[dict[str, Any]] = []
    for k in bucket_keys:
        bucket = buckets[k]
        take = targets[k]
        if take == 0:
            continue
        sampled.extend(rng.sample(bucket, take))

    # Top up if rounding came up short.
    if len(sampled) < n:
        seen_ids = {id(r) for r in sampled}
        leftover = [r for r in rows if id(r) not in seen_ids]
        rng.shuffle(leftover)
        sampled.extend(leftover[: n - len(sampled)])

    # Trim if we somehow over-shot (defensive).
    sampled = sampled[:n]
    rng.shuffle(sampled)
    return sampled


def _build_source_a(seed: int) -> list[dict[str, Any]]:
    """Materialise Source A (MedMCQA stems recast as open-ended)."""
    ds = load_dataset(MEDMCQA_REPO, split=MEDMCQA_SPLIT)
    # Keep only single-answer items where the gold option index is valid; the
    # validation split is consistently labelled but we filter defensively.
    # ``dict(r)`` materialises each HF row into a plain dict so the rest of
    # this function can use ``row.get(...)`` without tripping the union
    # return type of ``load_dataset``.
    rows: list[dict[str, Any]] = []
    for r in ds:
        row: dict[str, Any] = dict(r)
        cop = row.get("cop")
        if cop is None or cop < 0 or cop > 3:
            continue
        rows.append(row)

    sampled = _stratified_sample(rows, "subject_name", SOURCE_A_SIZE, seed)

    out: list[dict[str, Any]] = []
    for idx, row in enumerate(sampled, start=1):
        opts = [
            str(row.get("opa", "")),
            str(row.get("opb", "")),
            str(row.get("opc", "")),
            str(row.get("opd", "")),
        ]
        cop = int(row["cop"])
        gold = opts[cop].strip()
        subject = str(row.get("subject_name") or "general").strip().lower()
        record: dict[str, Any] = {
            "prompt_id": f"srcA_{idx:05d}",
            "source": "medmcqa_open",
            "subject": subject,
            "text": _recast_stem(str(row["question"])),
            "gold_answer": gold,
            "tags": [
                "medmcqa",
                f"gold_letter:{MCQ_LETTERS[cop]}",
                f"subject:{subject}",
            ],
            "_provenance": {
                "medmcqa_id": str(row.get("id", "")),
                "original_stem": str(row["question"]).strip(),
                "options": opts,
            },
        }
        out.append(record)
    return out


def _build_source_b(seed: int) -> list[dict[str, Any]]:
    """Materialise Source B (MedQA-USMLE vignettes)."""
    ds = load_dataset(MEDQA_REPO, split=MEDQA_SPLIT)
    rows: list[dict[str, Any]] = [dict(r) for r in ds]
    rng = random.Random(seed + 1)  # +1 so seed reuse across sources doesn't alias
    rng.shuffle(rows)
    sampled = rows[:SOURCE_B_SIZE]

    out: list[dict[str, Any]] = []
    for idx, row in enumerate(sampled, start=1):
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        meta_info = str(row.get("meta_info") or "").strip().lower()
        # MedQA does not carry a structured subject; ``meta_info`` is the
        # USMLE step (e.g. "step1", "step2&3"). Use it as the subject tag.
        subject = meta_info or "usmle"
        record: dict[str, Any] = {
            "prompt_id": f"srcB_{idx:05d}",
            "source": "medqa",
            "subject": subject,
            "text": question,
            "gold_answer": answer,
            "tags": [
                "medqa",
                f"usmle:{meta_info or 'unknown'}",
            ],
            "_provenance": {
                "answer_idx": str(row.get("answer_idx", "")),
                "options": dict(row.get("options") or {}),
            },
        }
        out.append(record)
    return out


def build_bank(seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Build the full prompt bank from Sources A and B.

    Source C (curated) is appended later by separate tooling; the loader
    downstream is expected to ``json.loads`` each line of ``bank.jsonl`` and
    merge any additional ``source: curated`` entries seamlessly.
    """
    bank: list[dict[str, Any]] = []
    bank.extend(_build_source_a(seed=seed))
    bank.extend(_build_source_b(seed=seed))
    return bank


def write_bank(bank: Iterable[dict[str, Any]], path: Path) -> None:
    """Write ``bank`` to ``path`` as JSON-Lines (one record per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in bank:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def load_bank(path: Path) -> list[dict[str, Any]]:
    """Read a previously-written bank.jsonl back into a list of records."""
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write bank.jsonl (default: alongside this script).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for stratified sampling; documented in run manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bank = build_bank(seed=args.seed)
    write_bank(bank, args.output)
    print(f"Wrote {len(bank)} prompts to {args.output} (seed={args.seed}).")


if __name__ == "__main__":
    main()

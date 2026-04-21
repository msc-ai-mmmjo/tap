import random
import re


def mock_claim_confidence(claim_text: str) -> dict:
    """
    Generate a mock confidence score. This will be replaced
    with our OLMo model once ready for integration.
    Uses text heuristics to make it look somewhat realistic:
    - Claims with hedging words get lower scores
    - Claims with specific numbers/dosages get moderate scores
    - General medical knowledge claims get higher scores
    """
    hedging = any(
        w in claim_text.lower()
        for w in [
            "may",
            "might",
            "possibly",
            "consider",
            "unclear",
            "uncertain",
            "could",
            "potentially",
            "suggest",
        ]
    )
    has_numbers = bool(
        re.search(
            r"\d+\s*(mg|ml|mcg|kg|%|days|weeks|hours|minutes)", claim_text.lower()
        )
    )

    if hedging:
        score = round(random.uniform(0.45, 0.65), 2)
    elif has_numbers:
        score = round(random.uniform(0.60, 0.80), 2)
    else:
        score = round(random.uniform(0.75, 0.95), 2)

    if score >= 0.80:
        return {"confidence": score, "level": "high", "guidance": ""}
    elif score >= 0.65:
        return {
            "confidence": score,
            "level": "moderate",
            "guidance": "Verify with clinical reference",
        }
    else:
        return {
            "confidence": score,
            "level": "low",
            "guidance": "Cross-check with authoritative source before acting",
        }


def mock_security_status() -> dict:
    return {
        "certified": True,
        "tpa_budget": random.randint(30, 80),
        "detail": "Certified radius against targeted poisoning",
    }


def mock_robustness_status(prompt: str) -> dict:
    return {
        "passed": True,
        "detail": "No generated suffix flipped the response",
        "flagged_tokens": [],
    }


_SWAP_POOL: list[tuple[str, str]] = [
    ("mg", "mcg"),
    ("days", "weeks"),
    ("hours", "days"),
    ("morning", "evening"),
    ("is", "may be"),
    ("should", "might"),
    ("adult", "paediatric"),
    ("oral", "IV"),
]

_HEDGING_WORDS = (
    "may",
    "might",
    "possibly",
    "consider",
    "unclear",
    "uncertain",
    "could",
    "potentially",
    "suggest",
)
_DOSAGE_PATTERN = re.compile(
    r"\d+\s*(mg|ml|mcg|kg|%|days|weeks|hours|minutes)", re.IGNORECASE
)
_MCQ_CHOICE_PATTERN = re.compile(r"\b([A-D])\b")

_SWAP_COUNT_WEIGHTS = (0.40, 0.30, 0.15, 0.10, 0.05)
_ROBUSTNESS_SCORES = (0.0, 0.5, 1.0, 1.5, 2.0)
_ROBUSTNESS_SCORE_WEIGHTS = (0.02, 0.08, 0.15, 0.30, 0.45)
_PERTURBATION_CHOICES = ("prefix", "hedge", "truncate")


def _uncertainty_score(response_text: str) -> float:
    lowered = response_text.lower()
    if any(w in lowered for w in _HEDGING_WORDS):
        return round(random.uniform(0.45, 0.65), 2)
    if _DOSAGE_PATTERN.search(lowered):
        return round(random.uniform(0.60, 0.80), 2)
    return round(random.uniform(0.75, 0.95), 2)


def _mock_uncertainty(response_text: str, is_mcq: bool | None) -> dict:
    if not response_text.strip():
        return {"overall": 0.0}
    if is_mcq is True:
        return {"overall": round(random.uniform(0.62, 0.94), 2)}
    return {"overall": _uncertainty_score(response_text)}


def _mock_security(response_text: str) -> dict:
    tokens = response_text.split()
    if not tokens:
        return {"tokens": [], "resampled": []}
    k = random.choices(range(5), weights=_SWAP_COUNT_WEIGHTS, k=1)[0]
    k = min(k, len(tokens))
    indices = random.sample(range(len(tokens)), k) if k else []
    resampled = []
    for idx in sorted(indices):
        old_tok, new_tok = random.choice(_SWAP_POOL)
        resampled.append(
            {
                "index": idx,
                "old_token": old_tok,
                "new_token": new_tok,
                "severity": round(random.uniform(0.2, 0.9), 2),
            }
        )
    return {"tokens": tokens, "resampled": resampled}


def _perturb_response(response_text: str) -> str:
    choice = random.choice(_PERTURBATION_CHOICES)
    if choice == "prefix":
        return f"Under adversarial pressure, {response_text}"
    if choice == "hedge":
        return re.sub(r"\bis\b", "may be", response_text, count=1)
    parts = response_text.rsplit(".", 2)
    if len(parts) >= 2 and parts[0].strip():
        return parts[0].strip() + "."
    return response_text


def _mock_robustness(response_text: str, is_mcq: bool | None) -> dict:
    if is_mcq is True:
        match = _MCQ_CHOICE_PATTERN.search(response_text)
        original = match.group(1) if match else "A"
        flipped = random.random() < 0.2
        if flipped:
            others = [c for c in "ABCD" if c != original]
            attacked = random.choice(others)
            attacked_response = f"On reflection, I would go with {attacked}."
        else:
            attacked = original
            attacked_response = f"I would still go with {original}."
        return {
            "type": "mcq",
            "flipped": flipped,
            "original_choice": original,
            "attacked_choice": attacked,
            "attacked_response": attacked_response,
        }
    if not response_text.strip():
        return {
            "type": "nlp",
            "bidirectional_score": 2.0,
            "attacked_response": "",
        }
    score = random.choices(_ROBUSTNESS_SCORES, weights=_ROBUSTNESS_SCORE_WEIGHTS, k=1)[0]
    return {
        "type": "nlp",
        "bidirectional_score": score,
        "attacked_response": _perturb_response(response_text),
    }


def build_analysis(response_text: str, is_mcq: bool | None) -> dict:
    return {
        "uncertainty": _mock_uncertainty(response_text, is_mcq),
        "security": _mock_security(response_text),
        "robustness": _mock_robustness(response_text, is_mcq),
    }

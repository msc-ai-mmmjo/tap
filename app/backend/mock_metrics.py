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

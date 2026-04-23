"""Small dict-builders for the ``/api/analyse`` response payload.

Kept separate from ``server.py`` so the FastAPI module stays focused on
request orchestration and model lifecycle; these helpers depend on neither.
"""


def fallback_security() -> dict:
    return {"certified": None, "tokens": [], "resampled": []}


def poe_security(tokens: list[str], resampled: list[dict]) -> dict:
    return {"certified": True, "tokens": tokens, "resampled": resampled}


def fallback_uncertainty() -> dict:
    return {"overall": None}


def poe_uncertainty(p_correct: float | None) -> dict:
    return {"overall": p_correct}


def fallback_robustness() -> dict:
    return {"type": "unavailable"}

"""Small dict-builders for the ``/api/analyse`` response payload.

Kept separate from ``server.py`` so the FastAPI module stays focused on
request orchestration and model lifecycle; these helpers depend on neither.
"""


def fallback_security() -> dict:
    return {
        "certified": None,
        "tokens": [],
        "resampled": [],
        "token_entropies": [],
    }


def poe_security(
    tokens: list[str],
    resampled: list[dict],
    token_entropies: list[float],
    stability_radii: list[int],
    stability_margins: list[float],
) -> dict:
    # token_entropies is conceptually an uncertainty signal, not a security one.
    # We park it on the security payload because the heatmap already consumes
    # `tokens` from here, so co-locating avoids cross-payload plumbing for now.
    return {
        "certified": True,
        "tokens": tokens,
        "resampled": resampled,
        "token_entropies": token_entropies,
        "stability_radii": stability_radii,
        "stability_margins": stability_margins,
    }


def fallback_uncertainty() -> dict:
    return {"overall": None}


def poe_uncertainty(p_correct: float | None) -> dict:
    return {"overall": p_correct}


def fallback_robustness() -> dict:
    return {"type": "unavailable"}

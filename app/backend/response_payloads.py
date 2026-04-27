"""Small dict-builders for the ``/api/analyse`` response payload.

Kept separate from ``server.py`` so the FastAPI module stays focused on
request orchestration and model lifecycle; these helpers depend on neither.
"""


def fallback_security() -> dict:
    """
    Security payload for the HF-fallback path.

    No PoE verification is performed against the HF Inference API response,
    so ``certified`` is ``None`` (frontend distinguishes this from the
    ``True`` case) and the per-token arrays are empty.

    :returns: Security sub-payload with all PoE-only fields blanked.
    """
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
    """
    Security payload built from PoE generation outputs.

    ``token_entropies`` is conceptually an uncertainty signal, not a
    security one; we park it on the security payload because the heatmap
    already consumes ``tokens`` from here, so co-locating avoids
    cross-payload plumbing for now.

    :param tokens: Decoded token strings parallel to ``token_entropies`` /
        ``stability_radii`` / ``stability_margins``.
    :param resampled: Per-rejection records (see
        :func:`app.backend.hydra_inference.generate` for the schema).
    :param token_entropies: Verifier ensemble predictive entropy in nats.
    :param stability_radii: Minimum vote flips needed to dethrone the
        winning token, parallel to ``tokens``.
    :param stability_margins: Top-1 minus top-2 verifier-ensemble
        probability, parallel to ``tokens``.

    :returns: Security sub-payload with ``certified=True`` indicating the
        response went through PoE verification.
    """
    return {
        "certified": True,
        "tokens": tokens,
        "resampled": resampled,
        "token_entropies": token_entropies,
        "stability_radii": stability_radii,
        "stability_margins": stability_margins,
    }


def fallback_uncertainty() -> dict:
    """
    Uncertainty payload when no certainty signal is available.

    Used both on the HF-fallback path and on the NLP path when KLE
    computation raises (e.g. NLI scorer hiccup).

    :returns: ``{"overall": None}``.
    """
    return {"overall": None}


def poe_uncertainty(p_correct: float | None) -> dict:
    """
    Uncertainty payload for the PoE path.

    For MCQ the input is the uncertainty-head's ``p_correct`` scalar; for
    NLP the caller substitutes the KLE-derived certainty into the same
    field so the frontend can render either uniformly.

    :param p_correct: Certainty score in ``[0, 1]`` or ``None`` when the
        signal is unavailable for this prompt.

    :returns: ``{"overall": p_correct}``.
    """
    return {"overall": p_correct}


def fallback_robustness() -> dict:
    """
    Robustness payload when adversarial probing was skipped.

    Returned on the HF-fallback path and when the NLP path can't compute
    similarity because BERT is unavailable.

    :returns: ``{"type": "unavailable"}``.
    """
    return {"type": "unavailable"}

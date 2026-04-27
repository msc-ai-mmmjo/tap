"""
FastAPI entrypoint for the Trustworthy Answer Protocol (TAP) backend.

Wires together the four scoring stages exposed by ``/api/analyse``:

1. **Generation** -- :func:`app.backend.hydra_inference.generate` runs the Hydra
   PoE ensemble; if the ensemble is unavailable the request falls through to
   the HF Inference API via :func:`call_hf_model`.
2. **Security** -- per-token PoE acceptance / verifier-ensemble entropy /
   stability radii are returned by :func:`app.backend.hydra_inference.generate`
   and packaged via :func:`app.backend.response_payloads.poe_security`.
3. **Uncertainty** -- ``p_correct`` from the uncertainty head for MCQ; for NLP
   we run :data:`olmo_tap.constants.KLE_N_SAMPLES` extra samples and convert
   their NLI similarity matrix into a Kernel Language Entropy certainty score.
4. **Robustness** -- :func:`app.backend.hydra_inference.get_robustness` retries
   the prompt with each adversarial suffix in :data:`ADV_SUFFIXES` and reports
   how many flipped the answer.

The two heavyweight models (Hydra + ModernBERT-NLI) are loaded once during
the FastAPI lifespan and stashed in module-level dicts so request handlers
can grab them without re-loading. On Modal the ``@modal.enter()`` hook in
:mod:`app.backend.modal_app` preloads Hydra into the same dicts before the
ASGI app boots; the lifespan detects this and skips the duplicate load.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from pydantic import BaseModel
from transformers import TokenizersBackend

from app.backend.adversarial_suffixes import ADV_SUFFIXES, N_ADV_SUFFIXES
from app.backend.bert_inference import load_bert
from app.backend.claim_confidence import compute_claim_confidences
from app.backend.claim_splitter import decompose_into_claims
from app.backend.constants import HF_FALLBACK_MODEL as HF_MODEL, HF_TOKEN
from app.backend.hydra_inference import (
    generate,
    get_robustness,
    load_hydra,
    MODEL_NAME,
)
from app.backend.question_classifier import detect_mcq_bert
from app.backend.response_payloads import (
    fallback_robustness,
    fallback_security,
    fallback_uncertainty,
    poe_security,
    poe_uncertainty,
)
from kernel_entropy.entropy import kle_from_similarity, kle_to_certainty
from kernel_entropy.nli import ModernBERTScorer
from olmo_tap.constants import KLE_HEAT_KERNEL_T, KLE_N_SAMPLES
from olmo_tap.hydra import HydraTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

_models: dict[str, Any | None] = {}
_tokenizers: dict[str, TokenizersBackend | None] = {}
_device: str = "cuda"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan that loads (or re-uses preloaded) Hydra and BERT models.

    On Modal the ``@modal.enter()`` hook in :mod:`app.backend.modal_app` has
    already populated ``_models["hydra"]`` before this runs; the duplicate
    load would otherwise add ~30s of cold-start. BERT is always loaded here
    because Modal's preload only warms Hydra.

    :param app: FastAPI application instance (unused but required by the
        lifespan protocol).
    """
    global _device
    _device = os.getenv("DEVICE", "cuda")
    logger.info("Starting up - device=%s", _device)

    # Modal's @modal.enter() may have already preloaded; skip to avoid a ~30s double-load.
    if "hydra" not in _models:
        _models["hydra"], _tokenizers["hydra"] = load_hydra(device=_device)
        if _models["hydra"] is None:
            logger.warning("Hydra unavailable; requests will fall back to HF API")
    else:
        logger.info("Hydra already preloaded; skipping lifespan load")

    if "bert" not in _models:
        _models["bert"], _tokenizers["bert"] = load_bert(device=_device)
        if _models["bert"] is None:
            logger.warning("BERT unavailable; NLI-based metrics will be skipped")
    else:
        logger.info("BERT already preloaded; skipping lifespan load")

    yield

    logger.info("Shutting down")
    _models.clear()
    _tokenizers.clear()


app = FastAPI(title="Trustworthy Answer Protocol - API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://tap-al9.pages.dev"],
    # Cloudflare Pages preview/PR deployments: <hash-or-branch>.tap-al9.pages.dev
    allow_origin_regex=r"^https://[a-z0-9-]+\.tap-al9\.pages\.dev$",
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    """
    Single chat-completions message.

    :param role: One of ``"system"``, ``"user"``, ``"assistant"`` (matches the
        OpenAI / HF chat-completions schema).
    :param content: Raw text content for that role.
    """

    role: str
    content: str


class ChatRequest(BaseModel):
    """
    Request body for :func:`analyse`.

    :param messages: Multi-turn chat history, oldest message first. The last
        element must have ``role == "user"`` and is the prompt that gets
        scored for uncertainty / security / robustness.
    """

    messages: list[Message]


def call_hf_model(messages: list[dict]) -> str:
    """Call the HF Inference API as a fallback when Hydra is unavailable or bypassed.

    Used when ``hf=true`` is passed to ``/api/analyse`` or when ``load_hydra``
    failed at lifespan startup. No PoE verification is available in this path;
    the security payload from the caller reflects that with ``certified=None``.
    """
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set")
    client = InferenceClient(HF_MODEL, token=HF_TOKEN)
    response = client.chat_completion(messages, max_tokens=500)
    return response.choices[0].message.content or ""


def _classify_mcq(last_user_msg: str) -> bool | None:
    """
    Classify the latest user message as MCQ or open-ended via BERT NLI.

    Wraps :func:`app.backend.question_classifier.detect_mcq_bert` and returns
    ``None`` when BERT failed to load at startup so the caller can degrade
    gracefully (no MCQ system prompt, NLP code paths only).

    :param last_user_msg: Raw text of the most recent user turn.

    :returns: ``True`` if multiple-choice, ``False`` if open-ended,
        ``None`` if BERT is unavailable.
    """
    bert_model = _models.get("bert")
    bert_tokenizer = _tokenizers.get("bert")
    if bert_model is None or bert_tokenizer is None:
        return None
    return detect_mcq_bert(bert_model, bert_tokenizer, last_user_msg, device=_device)


@app.post("/api/analyse")
async def analyse(request: ChatRequest, hf: bool = False):
    """
    Score a chat prompt for security, uncertainty, robustness and per-claim
    confidence.

    Generation runs through the Hydra PoE ensemble unless ``hf=True`` is
    passed (or the ensemble failed to load), in which case the HF Inference
    API is used as a fallback and security / uncertainty / robustness are
    returned as ``unavailable``-style payloads.

    The claim ledger is independent of the generation backend: it always
    decomposes ``raw_response`` and scores each claim with NLI self-entailment
    when BERT is available. KLE-based uncertainty for NLP queries is computed
    here (not inside ``generate``) because it requires :data:`KLE_N_SAMPLES`
    extra forward passes.

    :param request: Chat history; the last user turn is the prompt.
    :param hf: Force the HF Inference API path even when Hydra is healthy.
        Useful for A/B comparisons against the unverified baseline.

    :returns: Dict with keys ``claims``, ``overall_confidence``,
        ``uncertainty``, ``security``, ``robustness``, ``raw_response``,
        ``model``, ``is_mcq``. See
        :mod:`app.backend.response_payloads` for the security/uncertainty/
        robustness sub-schemas.
    """
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    latest_user_msg = messages[-1]["content"]
    logger.info("Latest user message: %s", latest_user_msg)

    is_mcq = _classify_mcq(latest_user_msg)
    logger.info("BERT MCQ classification: %s", is_mcq)

    hydra: HydraTransformer | None = _models.get("hydra")
    hydra_tokenizer: TokenizersBackend | None = _tokenizers.get("hydra")

    if hf or hydra is None or hydra_tokenizer is None:
        model_name = HF_MODEL
        raw_response = call_hf_model(messages)
        security = fallback_security()
        uncertainty = fallback_uncertainty()
        robustness = fallback_robustness()
    else:
        model_name = MODEL_NAME
        (
            raw_response,
            tokens,
            resampled,
            token_entropies,
            p_correct,
            stability_radii,
            stability_margins,
        ) = generate(
            hydra,
            hydra_tokenizer,
            messages,
            is_mcq=bool(is_mcq),
            device=_device,
        )
        security = poe_security(
            tokens, resampled, token_entropies, stability_radii, stability_margins
        )
        uncertainty = poe_uncertainty(p_correct)

        bert_model = _models.get("bert")
        bert_tokenizer = _tokenizers.get("bert")

        # Uncertainty for NLP
        if not is_mcq and bert_model is not None and bert_tokenizer is not None:
            try:
                kle_responses: list[str] = []
                for _ in range(KLE_N_SAMPLES):
                    raw, _t, _r, _e, _p, _, _ = generate(
                        hydra,
                        hydra_tokenizer,
                        messages,
                        is_mcq=False,
                        device=_device,
                    )
                    kle_responses.append(raw)

                W = ModernBERTScorer(
                    kle_responses,
                    model=bert_model,
                    tokenizer=bert_tokenizer,
                ).compute()
                entropy = kle_from_similarity(W, t=KLE_HEAT_KERNEL_T)  # type: ignore[arg-type]
                certainty = kle_to_certainty(entropy, KLE_N_SAMPLES)
                uncertainty = poe_uncertainty(certainty)
            except Exception:
                logger.exception("KLE computation failed; falling back")
                uncertainty = fallback_uncertainty()

        # Robustness
        if not is_mcq and (bert_model is None or bert_tokenizer is None):
            robustness = fallback_robustness()
        else:
            robustness = get_robustness(
                hydra,
                hydra_tokenizer,
                list(messages),
                original_resp=raw_response,
                original_tokens=tokens,
                is_mcq=bool(is_mcq),
                adv_suffix_bank=ADV_SUFFIXES[:N_ADV_SUFFIXES],
                bert_model=bert_model,
                bert_tokenizer=bert_tokenizer,
                device=_device,
            )

    logger.info("Generation complete (%d chars)", len(raw_response))

    bert_model = _models.get("bert")
    bert_tokenizer = _tokenizers.get("bert")

    claims: list[dict] = []
    overall: float | None = None
    if bert_model is not None and bert_tokenizer is not None:
        try:
            claims_text = decompose_into_claims(raw_response)
            metrics_list = compute_claim_confidences(
                raw_response, claims_text, bert_model, bert_tokenizer
            )
            claims = [
                {
                    "text": text,
                    "confidence": m["confidence"],
                    "confidence_level": m["level"],
                    "guidance": m["guidance"],
                }
                for text, m in zip(claims_text, metrics_list)
            ]
            if metrics_list:
                overall = round(
                    sum(m["confidence"] for m in metrics_list) / len(metrics_list), 2
                )
        except Exception:
            logger.exception("Claim ledger unavailable; returning empty claims")
            claims = []
            overall = None

    return {
        "claims": claims,
        "overall_confidence": overall,
        "uncertainty": uncertainty,
        "security": security,
        "robustness": robustness,
        "raw_response": raw_response,
        "model": model_name,
        "is_mcq": is_mcq,
    }


@app.get("/api/health")
async def health():
    """
    Lightweight liveness probe used by Cloudflare Pages, uptime checks and
    Modal's health monitor.

    :returns: ``{"status": "ok"}``. Does not touch model state, so a 200 here
        only means the ASGI process is up; readiness for Hydra requests is
        implicit in successful ``/api/analyse`` calls.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    # Local smoke test:
    #   pixi run -e cuda python -m app.backend.server
    # Hits the in-process ASGI app via TestClient so no uvicorn / port is
    # needed. The lifespan still fires, so this exercises the same model-
    # loading path as a real ``modal serve`` deployment.
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        health_resp = client.get("/api/health")
        print("Health:", health_resp.json())

        analyse_resp = client.post(
            "/api/analyse",
            json={
                "messages": [
                    {"role": "user", "content": "Is paracetamol safe in pregnancy?"}
                ]
            },
        )
        body = analyse_resp.json()
        print("Model:", body["model"])
        print("Is MCQ:", body["is_mcq"])
        print("Response:", body["raw_response"][:200])

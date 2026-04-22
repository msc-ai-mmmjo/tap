import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from pydantic import BaseModel
from transformers import TokenizersBackend

from app.backend.bert_inference import load_bert
from app.backend.claim_splitter import decompose_into_claims
from app.backend.constants import HF_FALLBACK_MODEL as HF_MODEL, HF_TOKEN
from app.backend.hydra_inference import generate, load_hydra, MODEL_NAME
from app.backend.mock_metrics import (
    mock_claim_confidence,
    mock_robustness_status,
)
from app.backend.question_classifier import detect_mcq_bert
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
            logger.warning("BERT unavailable; MCQ classification disabled")
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
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


def call_hf_model(messages: list[dict]) -> str:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set")
    client = InferenceClient(HF_MODEL, token=HF_TOKEN)
    response = client.chat_completion(messages, max_tokens=500)
    return response.choices[0].message.content or ""


def _classify_mcq(last_user_msg: str) -> bool | None:
    bert_model = _models.get("bert")
    bert_tokenizer = _tokenizers.get("bert")
    if bert_model is None or bert_tokenizer is None:
        return None
    return detect_mcq_bert(bert_model, bert_tokenizer, last_user_msg, device=_device)


def _fallback_security() -> dict:
    return {"certified": None, "tokens": [], "resampled": []}


def _poe_security(tokens: list[str], resampled: list[dict]) -> dict:
    return {"certified": True, "tokens": tokens, "resampled": resampled}


@app.post("/api/analyse")
async def analyse(request: ChatRequest, hf: bool = False):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    logger.info("Latest user message: %s", messages[-1]["content"])

    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    is_mcq = _classify_mcq(last_user_msg)
    logger.info("BERT MCQ classification: %s", is_mcq)

    hydra: HydraTransformer | None = _models.get("hydra")
    hydra_tokenizer: TokenizersBackend | None = _tokenizers.get("hydra")

    if hf or hydra is None or hydra_tokenizer is None:
        model_name = HF_MODEL
        raw_response = call_hf_model(messages)
        security = _fallback_security()
    else:
        model_name = MODEL_NAME
        raw_response, tokens, resampled = generate(
            hydra,
            hydra_tokenizer,
            messages,
            is_mcq=bool(is_mcq),
            device=_device,
        )
        security = _poe_security(tokens, resampled)

    logger.info("Generation complete (%d chars)", len(raw_response))

    claims_text = decompose_into_claims(raw_response)
    claims = []
    scores = []
    for text in claims_text:
        metrics = mock_claim_confidence(text)
        scores.append(metrics["confidence"])
        claims.append(
            {
                "text": text,
                "confidence": metrics["confidence"],
                "confidence_level": metrics["level"],
                "guidance": metrics["guidance"],
            }
        )

    overall = round(sum(scores) / len(scores), 2) if scores else 0.0

    return {
        "claims": claims,
        "overall_confidence": overall,
        "security": security,
        "robustness": mock_robustness_status(last_user_msg),
        "raw_response": raw_response,
        "model": model_name,
        "is_mcq": is_mcq,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}

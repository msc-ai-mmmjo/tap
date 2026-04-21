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
from app.backend.constants import HF_TOKEN, BERT_MCQ_DETECTION
from app.backend.question_classifier import detect_mcq_bert
from app.backend.hydra_inference import generate, load_hydra, MODEL_NAME
from app.backend.mock_metrics import (
    mock_claim_confidence,
    mock_robustness_status,
    mock_security_status,
)
from gradio_demo.constants import MODEL as HF_MODEL
from olmo_tap.constants import MAX_NEW_TOKENS
from olmo_tap.hydra import HydraTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

_models: dict[str, Any | None] = {}
_tokenizers: dict[str, TokenizersBackend | None] = {}
_device: str = "cuda"

_important_tokens: dict[str, int] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _device
    _device = os.getenv("DEVICE", "cuda")
    logger.info("Starting up — device=%s", _device)

    _models["hydra"], _tokenizers["hydra"] = load_hydra(device=_device)
    if _models["hydra"] is None:
        logger.warning("Hydra unavailable; requests will fall back to HF API")
    if _tokenizers["hydra"] is not None:
        for token in ["A", "B", "C", "D"]:
            _important_tokens[token] = _tokenizers["hydra"].encode(
                token, add_special_tokens=False
            )[0]

    _models["bert"], _tokenizers["bert"] = load_bert(device=_device)
    if _models["bert"] is None:
        logger.warning("BERT unavailable; NLI-based metrics will be skipped")

    yield

    logger.info("Shutting down")
    _models.clear()
    _tokenizers.clear()


app = FastAPI(title="Trustworthy Answer Protocol — API", lifespan=lifespan)
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
    """Call the HF model using the same approach as the Gradio app."""
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set")

    client = InferenceClient(HF_MODEL, token=HF_TOKEN)
    response = client.chat_completion(messages, max_tokens=500)
    return response.choices[0].message.content or ""


@app.post("/api/analyse")
async def analyse(request: ChatRequest, hf: bool = False):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    robustness = mock_robustness_status(messages[-1]["content"])
    logger.info("Latest user message: %s", messages[-1]["content"])

    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )

    hydra: HydraTransformer | None = _models.get("hydra")
    hydra_tokenizer: TokenizersBackend | None = _tokenizers.get("hydra")

    is_mcq = None
    if hf or hydra is None or hydra_tokenizer is None:
        raw_response = call_hf_model(messages)
        model = HF_MODEL
    else:
        raw_response, is_mcq = generate(
            hydra,
            hydra_tokenizer,
            messages,
            MAX_NEW_TOKENS,
            device=_device,
            important_token_ids=_important_tokens,
        )
        model = MODEL_NAME
    logger.info("Generation complete (%d chars)", len(raw_response))

    if not BERT_MCQ_DETECTION:
        logger.info("Hydra MCQ classification: %s", is_mcq)
    else:
        bert_model = _models.get("bert")
        bert_tokenizer = _tokenizers.get("bert")
        if bert_model is not None and bert_tokenizer is not None:
            is_mcq = detect_mcq_bert(
                bert_model, bert_tokenizer, last_user_msg, device=_device
            )
            logger.info("BERT MCQ classification: %s", is_mcq)

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
        "security": mock_security_status(),
        "robustness": robustness,
        "raw_response": raw_response,
        "model": model,
        "is_mcq": is_mcq,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}

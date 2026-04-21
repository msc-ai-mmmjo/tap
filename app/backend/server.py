import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from pydantic import BaseModel
from transformers import TokenizersBackend

from app.backend.hydra_inference import generate, load_model, MODEL_NAME
from app.backend.mock_metrics import (
    mock_claim_confidence,
    mock_robustness_status,
    mock_security_status,
)
from app.backend.constants import HF_TOKEN
from gradio_demo.constants import MODEL as HF_MODEL
from olmo_tap.constants import MAX_NEW_TOKENS
from olmo_tap.hydra import HydraTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

_model: HydraTransformer | None = None
_tokenizer: TokenizersBackend | None = None
_device: str = "cuda"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _device
    _device = os.getenv("DEVICE", "cuda")
    logger.info("Starting up — device=%s", _device)
    _model, _tokenizer = load_model(device=_device)
    if _model is None:
        logger.warning("Model unavailable; requests will fall back to HF API")
    yield
    logger.info("Shutting down")
    _model = None
    _tokenizer = None


app = FastAPI(title="Trustworthy Answer Protocol — API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://tap-al9.pages.dev"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


def decompose_into_claims(text: str) -> list[str]:
    """Split response into individual assertions. Filters sentences shorter than 20 chars."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not claims:
        claims = [text.strip()]
    return claims


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

    if hf or _model is None or _tokenizer is None:
        raw_response = call_hf_model(messages)
        model = HF_MODEL
    else:
        raw_response = generate(
            _model, _tokenizer, messages, MAX_NEW_TOKENS, device=_device
        )
        model = MODEL_NAME
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
        "security": mock_security_status(),
        "robustness": robustness,
        "raw_response": raw_response,
        "model": model,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}

import os
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from pydantic import BaseModel

from mock_metrics import mock_claim_confidence, mock_robustness_status, mock_security_status

_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

app = FastAPI(title="Trustworthy Answer Protocol — Demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


def decompose_into_claims(text: str) -> list[str]:
    """
    Split model response into individual clinical assertions.
    Simple sentence-level splitting for now.
    Filter out very short sentences (likely not claims).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not claims:
        claims = [text.strip()]
    return claims


def call_hf_model(messages: list[dict]) -> str:
    """Call the HF model using the same approach as the Gradio app."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    client = InferenceClient(MODEL, token=hf_token)
    response = client.chat_completion(messages, max_tokens=500)
    return response.choices[0].message.content


@app.post("/api/analyse")
async def analyse(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    robustness = mock_robustness_status(messages[-1]["content"])

    raw_response = call_hf_model(messages)

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
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}

import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import InferenceClient
from pydantic import BaseModel

from app.backend.constants import HF_TOKEN
from app.backend.mock_metrics import (
    mock_claim_confidence,
    mock_robustness_status,
    mock_security_status,
)
from gradio_demo.constants import MODEL

app = FastAPI(title="Trustworthy Answer Protocol — API")
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


def decompose_into_claims(text: str) -> list[dict]:
    """
    Split model response into clinical assertions with character offsets
    into the original text, so the frontend can highlight claims inline.

    Placeholder implementation: naive sentence-level splitting via regex.
    The real decomposer (FActScore / SAFE / prompted LLM) will return
    sub-sentence atomic propositions, may paraphrase the text, and may
    emit overlapping or non-literal spans. See PR discussion for the
    contract the real decomposer should honour (text vs evidence span).
    """
    claims: list[dict] = []
    for m in re.finditer(r"\S[^.!?]*[.!?]+", text):
        span_text = m.group().strip()
        if len(span_text) > 20:
            start = m.start()
            claims.append(
                {"text": span_text, "start": start, "end": start + len(span_text)}
            )
    if not claims and text.strip():
        start = len(text) - len(text.lstrip())
        stripped = text.strip()
        claims.append({"text": stripped, "start": start, "end": start + len(stripped)})
    return claims


def call_hf_model(messages: list[dict]) -> str:
    """Call the HF model using the same approach as the Gradio app."""
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set")

    client = InferenceClient(MODEL, token=HF_TOKEN)
    response = client.chat_completion(messages, max_tokens=500)
    return response.choices[0].message.content or ""


@app.post("/api/analyse")
async def analyse(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    robustness = mock_robustness_status(messages[-1]["content"])

    raw_response = call_hf_model(messages)

    decomposed = decompose_into_claims(raw_response)
    claims = []
    scores = []
    for c in decomposed:
        metrics = mock_claim_confidence(c["text"])
        scores.append(metrics["confidence"])
        claims.append(
            {
                "text": c["text"],
                "start": c["start"],
                "end": c["end"],
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
        "model": MODEL,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok"}

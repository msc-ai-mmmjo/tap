"""Atomic claim decomposition for /api/analyse.

Primary path: prompt the HF generation model to break its response into
atomic, self-contained factual claims. This mirrors the decomposition
step of fact-checking pipelines such as FActScore (Min et al. 2023).

Fallback: NLTK sentence segmentation when the LLM path fails (missing
token, API error, or unparseable output). Sentences are a coarser unit
but keep the endpoint functional instead of returning nothing.
"""

import json

import nltk
from huggingface_hub import InferenceClient
from nltk.tokenize import sent_tokenize

from app.backend.constants import HF_TOKEN
from gradio_demo.constants import MODEL

_SYSTEM_PROMPT = """\
You decompose text responses into atomic, self-contained factual claims,
returned as a JSON array of strings.

Rules:
- Each claim expresses a single factual assertion.
- Each claim is self-contained: resolve pronouns and include the subject
  and any context needed to understand the claim standalone.
- Use only information present in the input; do not add new facts.
- Split compound sentences (e.g., dose + frequency + duration become
  separate claims).
- Preserve numbers, units, and proper nouns exactly.
- Skip greetings, transitions, and pure filler.
- If the input contains no verifiable claims, return [].

Output: a JSON array of strings. No prose, no code fences.
"""

_USER_PROMPT = 'Decompose this response:\n\n"""\n{text}\n"""'
_MAX_DECOMPOSE_TOKENS = 600


def _parse_json_array(raw: str) -> list[str] | None:
    """Parse a JSON array of non-empty strings from the LLM's output.

    `json.loads` alone fails when the model wraps the array in prose or
    a markdown fence, so the outermost `[...]` is extracted before
    parsing. Returns None if no array is present or the result is not a
    list of strings; the caller treats None as failure and falls back.
    """
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return [s.strip() for s in parsed if isinstance(s, str) and s.strip()]


def _llm_decompose(text: str) -> list[str] | None:
    """Ask the HF model for an atomic-claim decomposition.

    Returns None on any failure (missing token, API error, unparseable
    output) so the caller can fall back to sentence segmentation.
    """
    if not HF_TOKEN:
        return None
    try:
        client = InferenceClient(MODEL, token=HF_TOKEN)
        response = client.chat_completion(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_PROMPT.format(text=text)},
            ],
            max_tokens=_MAX_DECOMPOSE_TOKENS,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
    except Exception:
        return None
    claims = _parse_json_array(raw)
    return claims if claims else None


def _nltk_sentences(text: str) -> list[str]:
    """Segment text into sentences; download punkt_tab on first use.

    The tokenizer data is a one-time ~10 MB download, cached by NLTK in
    the user's home directory. If the download itself fails (e.g. no
    network), we give up splitting and return the whole text as one unit.
    """
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        if not nltk.download("punkt_tab", quiet=True):
            return [text]
        sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()] or [text]


def decompose_into_claims(text: str) -> list[str]:
    """Split an LLM response into atomic, self-contained claims.

    Tries LLM-based atomic decomposition first; on failure, falls back to
    NLTK sentence segmentation.
    """
    text = text.strip()
    if not text:
        return []
    claims = _llm_decompose(text)
    if claims:
        return claims
    return _nltk_sentences(text)

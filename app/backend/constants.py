"""
Backend-only environment configuration.

Reads :envvar:`HF_TOKEN` (HF Inference API + claim decomposition) and
:envvar:`HF_CACHE_DIR` (mounted HF cache for ModernBERT-NLI) from the
process environment, with a ``.env`` fallback for local development.
:data:`HF_FALLBACK_MODEL` is the model used both as the generation
fallback when Hydra is unavailable and as the LLM that performs atomic
claim decomposition in :mod:`app.backend.claim_splitter`.

Lives here rather than under :mod:`olmo_tap` so the research package has
no dependency on FastAPI / HF API plumbing.
"""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")

# HF Inference API model used as the fallback generator and for atomic claim
# decomposition.
HF_FALLBACK_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

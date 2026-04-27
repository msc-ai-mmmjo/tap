"""
ModernBERT-large-NLI loader shared by every NLI-based scorer in the backend.

The same model is reused for three distinct jobs at request time:

- **MCQ classification** (:func:`app.backend.question_classifier.detect_mcq_bert`)
- **Per-claim self-entailment** (:func:`app.backend.claim_confidence.compute_claim_confidences`)
- **Kernel Language Entropy + robustness similarity matrices**
  (:class:`kernel_entropy.nli.ModernBERTScorer`)

Loading is wrapped in try/except so a missing :data:`HF_CACHE_DIR` or HF
network blip downgrades to ``(None, None)`` rather than crashing the
FastAPI lifespan.
"""

import logging
import time

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TokenizersBackend,
)

from app.backend.constants import HF_CACHE_DIR

logger = logging.getLogger(__name__)

MODEL_ID = "tasksource/ModernBERT-large-nli"


def load_bert(device: str = "cuda"):
    """
    Load ModernBERT-large-NLI from the Modal-mounted HF cache.

    The Modal volume is populated once by
    :func:`app.backend.modal_app.download_weights`; locally
    :data:`HF_CACHE_DIR` may point at the user's HF cache.

    :param device: Torch device for the loaded model.

    :returns: ``(model, tokenizer)`` on success, ``(None, None)`` if the
        cache is missing or the download fails. The FastAPI lifespan logs
        and continues in degraded mode when this returns ``None``.
    """
    logger.info("Loading BERT for NLI-based metrics. Using cache dir: %s", HF_CACHE_DIR)
    t0 = time.perf_counter()

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR)
        if not isinstance(tokenizer, TokenizersBackend):
            logger.error("Tokenizer is not a TokenizersBackend; aborting model load")
            return None, None

        model = (
            AutoModelForSequenceClassification.from_pretrained(
                MODEL_ID, cache_dir=HF_CACHE_DIR
            )
            .to(device)
            .eval()
        )
    except Exception as e:
        logger.error("BERT unavailable: %s", e)
        return None, None

    logger.info("BERT loaded in %.1fs", time.perf_counter() - t0)
    return model, tokenizer

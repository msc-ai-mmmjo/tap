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
    t0 = time.perf_counter()

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_DIR)
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

    if not isinstance(tokenizer, TokenizersBackend):
        logger.error("Tokenizer is not a TokenizersBackend; aborting model load")
        return None, None

    logger.info("BERT loaded in %.1fs", time.perf_counter() - t0)
    return model, tokenizer

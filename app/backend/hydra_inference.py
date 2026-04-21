import logging
import time
from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase, TokenizersBackend

from olmo_tap.constants import MCQ_LETTERS, NLP_MAX_NEW_TOKENS, WEIGHTS_DIR
from olmo_tap.hydra import HydraTransformer
from olmo_tap.inference.loading_weights import load_ensemble
from olmo_tap.inference.mcq import poe_mcq_predict
from olmo_tap.inference.poe import poe_generate_with_cache

logger = logging.getLogger(__name__)

MODEL_NAME = "Hydra"


def load_hydra(
    device: str = "cuda",
) -> tuple[HydraTransformer, TokenizersBackend] | tuple[None, None]:
    """Load the 9-head hydra with prod security + robustness LoRAs merged."""
    t0 = time.perf_counter()

    if not WEIGHTS_DIR:
        logger.warning("WEIGHTS_DIR not set; skipping model load")
        return None, None

    logger.info("Loading tokenizer from %s", WEIGHTS_DIR)
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    if not isinstance(tokenizer, TokenizersBackend):
        logger.error("Tokenizer is not a TokenizersBackend; aborting model load")
        return None, None

    logger.info("Building ensemble on device=%s", device)
    try:
        model, _n_heads = load_ensemble()
    except Exception as e:
        logger.error("Error loading ensemble: %s", e)
        return None, None

    logger.info("Model ready -- setup took %.2fs", time.perf_counter() - t0)
    return model, tokenizer


def generate(
    model: HydraTransformer,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    is_mcq: bool,
    device: str = "cuda",
) -> tuple[str, list[dict]]:
    """Generate a PoE response.

    :returns: ``(raw_response, flagged_tokens)``. For MCQ, ``raw_response`` is a
        single letter and ``flagged_tokens`` is empty. For NLP, ``raw_response``
        is the PoE output and ``flagged_tokens`` is a list of
        ``{start, end, original, replacement}`` dicts.
    """
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    n_heads = len(model.heads)

    t0 = time.perf_counter()

    if is_mcq:
        abcd_token_ids = [
            tokenizer.encode(letter, add_special_tokens=False)[0]
            for letter in MCQ_LETTERS
        ]
        letter = poe_mcq_predict(
            model, tokenizer, chat_prompt, abcd_token_ids, device=device
        )
        logger.info(
            "MCQ PoE prediction: %s (%.2fs)", letter, time.perf_counter() - t0
        )
        return letter, []

    output_parts, original_tokens, resampled_idxs = poe_generate_with_cache(
        model,
        tokenizer,
        prompt_text="",  # ignored when messages is provided
        n_heads=n_heads,
        max_new_tokens=NLP_MAX_NEW_TOKENS,
        messages=messages,
    )
    raw_response, flagged_tokens = _spans_from_poe_output(
        tokenizer, output_parts, original_tokens, resampled_idxs
    )
    logger.info(
        "NLP PoE generation: %d chars, %d resamples (%.2fs)",
        len(raw_response),
        len(flagged_tokens),
        time.perf_counter() - t0,
    )
    return raw_response, flagged_tokens


def _spans_from_poe_output(
    tokenizer: PreTrainedTokenizerBase,
    output_parts: list[str],
    original_tokens: list[str],
    resampled_idxs: list[int],
) -> tuple[str, list[dict]]:
    """Convert poe.py's token-indexed output into char-offset spans.

    ``output_parts[0]`` is the chat-templated input (stripped). Subsequent
    entries are decoded single tokens from the generation. ``resampled_idxs``
    indexes into ``output_parts``; ``original_tokens[j]`` is the rejected draft
    token at ``resampled_idxs[j]``. Any trailing EOS is stripped from both
    ``raw_response`` and any spans that would land on it.
    """
    eos_id = tokenizer.eos_token_id
    eos_str = cast(str, tokenizer.decode([eos_id])) if eos_id is not None else ""

    parts = list(output_parts[1:])
    while parts and eos_str and parts[-1] == eos_str:
        parts.pop()

    spans: list[dict] = []
    offset = 0
    for i, part in enumerate(parts):
        orig_idx = i + 1  # index in the original output_parts
        if orig_idx in resampled_idxs:
            j = resampled_idxs.index(orig_idx)
            spans.append(
                {
                    "start": offset,
                    "end": offset + len(part),
                    "original": original_tokens[j],
                    "replacement": part,
                }
            )
        offset += len(part)

    raw_response = "".join(parts)
    return raw_response, spans

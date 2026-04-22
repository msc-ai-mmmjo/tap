import logging
import time
from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase, TokenizersBackend

from olmo_tap.constants import MAX_NEW_TOKENS, WEIGHTS_DIR
from olmo_tap.hydra import HydraTransformer
from olmo_tap.inference.loading_weights import load_ensemble
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
) -> tuple[str, list[str], list[dict]]:
    """Generate a PoE response via speculative verification.

    Single inference path for both MCQ and NLP prompts, per the response
    aggregation doc ("always PoE"). ``is_mcq`` does not alter generation today;
    it is kept as the routing flag for downstream robustness and uncertainty
    metrics.

    :returns: ``(raw_response, tokens, resampled)`` where ``tokens`` is the
        stripped token stream and ``resampled`` lists PoE rejections with
        token-level indices.
    """
    del is_mcq  # reserved for downstream metric routing; see docstring
    n_heads = len(model.heads)

    t0 = time.perf_counter()

    # TODO: poe_generate_with_cache hardcodes device="cuda"; the ``device`` arg
    # here is currently ignored.
    output_parts, original_tokens, resampled_idxs = poe_generate_with_cache(
        model,
        tokenizer,
        prompt_text="",
        n_heads=n_heads,
        max_new_tokens=MAX_NEW_TOKENS,
        messages=messages,
    )
    raw_response, tokens, resampled = _tokens_and_resamples_from_poe_output(
        tokenizer, output_parts, original_tokens, resampled_idxs
    )
    logger.info(
        "PoE generation: %d chars, %d/%d tokens resampled (%.2fs)",
        len(raw_response),
        len(resampled),
        len(tokens),
        time.perf_counter() - t0,
    )
    return raw_response, tokens, resampled


def _tokens_and_resamples_from_poe_output(
    tokenizer: PreTrainedTokenizerBase,
    output_parts: list[str],
    original_tokens: list[str],
    resampled_idxs: list[int],
) -> tuple[str, list[str], list[dict]]:
    """Convert poe.py's token-indexed output into (raw_response, tokens, resampled).

    ``output_parts[0]`` is the chat-templated input; subsequent entries are
    decoded single tokens from the generation. ``resampled_idxs`` indexes into
    ``output_parts``; ``original_tokens[j]`` is the rejected draft token at
    ``resampled_idxs[j]``. Trailing EOS entries are trimmed from both the
    emitted token stream and any resample records that would land on them.
    Each token's outer whitespace is stripped so the frontend can join tokens
    with a single space for display without double-spacing BPE continuations.
    """
    eos_id = tokenizer.eos_token_id
    eos_str = cast(str, tokenizer.decode([eos_id])) if eos_id is not None else ""

    parts = list(output_parts[1:])
    while parts and eos_str and parts[-1] == eos_str:
        parts.pop()

    raw_response = "".join(parts)
    tokens = [p.strip() for p in parts]

    resampled: list[dict] = []
    for j, orig_idx in enumerate(resampled_idxs):
        token_idx = orig_idx - 1
        if not 0 <= token_idx < len(parts):
            continue
        resampled.append(
            {
                "index": token_idx,
                "old_token": original_tokens[j].strip(),
                "new_token": parts[token_idx].strip(),
                # Placeholder until per-shard beta_h reliability weights land.
                "severity": 1.0,
            }
        )

    return raw_response, tokens, resampled

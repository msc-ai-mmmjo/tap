import logging
import time
from typing import Any, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase, TokenizersBackend

from kernel_entropy.nli import ModernBERTScorer
from olmo_tap.constants import MAX_NEW_TOKENS, MCQ_MAX_NEW_TOKENS, WEIGHTS_DIR
from olmo_tap.hydra import HydraTransformer
from olmo_tap.inference.loading_weights import load_ensemble
from olmo_tap.inference.poe import PoE

NLP_ROBUSTNESS_THRESHOLD = 1.0

logger = logging.getLogger(__name__)

MODEL_NAME = "Hydra"

# TODO Per Michele's comment, we should play with this to see if we can get option +
# some short explanation and refactor relevant inference pipeline e.g. robustness
# to reflect that. From some tests I performed if it was left to free flow it would
# just give dictionary definitions of each option.
# I also wonder what happens if it gets an MCQ with no letter? Unsure what would happen
MCQ_SYSTEM_PROMPT = (
    "Respond with only the letter of the correct option (e.g. A, B, C, or D). "
    "Do not add any explanation."
)

NLP_SYSTEM_PROMPT = (
    "You are a medical expert. "
    "Answer directly in at most 5 short sentences. "
    "No preamble, headers, lists, disclaimers, or restating the question. "
    "Do not tell the user to consult a professional. "
    "Put the final answer in the first sentence."
)


def load_hydra(
    device: str = "cuda",
) -> tuple[HydraTransformer, TokenizersBackend] | tuple[None, None]:
    """Load the 10-head hydra (9 LLM + 1 uncertainty) with prod security + robustness LoRAs merged."""
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
) -> tuple[str, list[str], list[dict], float | None]:
    """Generate a PoE response via speculative verification.

    Both MCQ and NLP prompts run through ``PoE.generate_with_cache``. When
    ``is_mcq`` is true, a short system nudge is prepended, the token budget is
    capped at ``MCQ_MAX_NEW_TOKENS`` so the model emits a bare letter, and the
    PoE class captures a witness hidden state on the first accepted/rejected
    token and returns a ``p_correct`` scalar from a dedicated uncertainty head.

    :returns: ``(raw_response, tokens, resampled, uncertainty)`` where
        ``uncertainty`` is a float ``p_correct`` for MCQ and ``None`` for NLP.
    """
    n_heads = len(model.heads)

    if is_mcq:
        messages = [{"role": "system", "content": MCQ_SYSTEM_PROMPT}, *messages]
        max_new_tokens = MCQ_MAX_NEW_TOKENS
    else:
        messages = [{"role": "system", "content": NLP_SYSTEM_PROMPT}, *messages]
        max_new_tokens = MAX_NEW_TOKENS

    t0 = time.perf_counter()

    # TODO: PoE hardcodes device="cuda" internally; the ``device`` arg here is
    # currently ignored.
    poe = PoE(
        model,
        tokenizer,
        n_llm_heads=n_heads - 1,
        max_new_tokens=max_new_tokens,
    )
    output_parts, original_tokens, resampled_idxs, uncertainty = (
        poe.generate_with_cache(
            prompt_text="",
            is_mcq=is_mcq,
            messages=messages,
        )
    )
    raw_response, tokens, resampled = _tokens_and_resamples_from_poe_output(
        tokenizer, output_parts, original_tokens, resampled_idxs
    )
    logger.info(
        "PoE generation: %d chars, %d/%d tokens resampled, uncertainty=%s (%.2fs)",
        len(raw_response),
        len(resampled),
        len(tokens),
        f"{uncertainty:.4f}" if uncertainty is not None else "n/a",
        time.perf_counter() - t0,
    )
    return raw_response, tokens, resampled, uncertainty


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


def get_robustness(
    model: HydraTransformer,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    original_resp: str,
    is_mcq: bool,
    adv_suffix_bank: list[str],
    bert_model: Any,
    bert_tokenizer: Any,
    device: str = "cuda",
) -> dict:
    """Return a robustness report by testing adversarial suffixes against the original response."""
    last_message = messages.pop()

    num_flipped = 0
    # (suffix, adv_resp) for MCQ; (score, suffix, adv_resp) for NLP
    mcq_first_flip: tuple[str, str] | None = None
    nlp_entries: list[tuple[float, str, str]] = []

    for suffix in adv_suffix_bank:
        logger.info("Testing adversarial suffix: %s", suffix)

        attack_msg = {"role": "user", "content": last_message["content"] + suffix}
        adv_prompt = messages + [attack_msg]
        adv_resp, _, _, _ = generate(model, tokenizer, adv_prompt, is_mcq, device)

        if is_mcq:
            if original_resp.strip().upper() != adv_resp.strip().upper():
                logger.info("Adversarial suffix '%s' caused MCQ answer change!", suffix)
                num_flipped += 1
                if mcq_first_flip is None:
                    mcq_first_flip = (suffix, adv_resp)
        else:
            scorer = ModernBERTScorer(
                [original_resp, adv_resp],
                model=bert_model,
                tokenizer=bert_tokenizer,
            )
            similarity_mat, raw_probs = scorer.compute(verbose=True)
            score = similarity_mat[0, 1].item()

            # TODO - could use `raw_probs` contents directly as finer-grained
            # NLI change measure instead of aggregated similarity score
            if score <= NLP_ROBUSTNESS_THRESHOLD:
                logger.info(
                    "Adversarial suffix '%s' caused significant NLP answer change!",
                    suffix,
                )
                num_flipped += 1
            nlp_entries.append((score, suffix, adv_resp))

    logger.info(
        "Robustness (%s): %d/%d flipped",
        "mcq" if is_mcq else "nlp",
        num_flipped,
        len(adv_suffix_bank),
    )

    # Worst-case entry for the adversarial preview panel:
    # NLP -> lowest NLI similarity (always returned); MCQ -> first flipped suffix, else None.
    worst_case: dict | None = None
    if is_mcq:
        if mcq_first_flip is not None:
            suffix, adv_resp = mcq_first_flip
            worst_case = {
                "suffix": suffix,
                "clean_response": original_resp,
                "adv_response": adv_resp,
                "flipped": True,
                "score": None,
            }
    elif nlp_entries:
        worst_score, worst_suffix, worst_adv_resp = min(nlp_entries, key=lambda e: e[0])
        worst_case = {
            "suffix": worst_suffix,
            "clean_response": original_resp,
            "adv_response": worst_adv_resp,
            "flipped": worst_score <= NLP_ROBUSTNESS_THRESHOLD,
            "score": worst_score,
        }

    return {
        "type": "mcq" if is_mcq else "nlp",
        "attempts": len(adv_suffix_bank),
        "flipped": num_flipped,
        "worst_case": worst_case,
    }

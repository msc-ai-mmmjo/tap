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

MCQ_SYSTEM_PROMPT = (
    "Start your answer with the chosen option, nothing before it."
    "- If the question lists lettered options (A, B, C, D), output just the letter."
    "- If the question is yes/no, output just 'yes' or 'no'."
    "- Otherwise, output just the option text exactly as listed in the question."
    "You may then add a short explanation if it helps."
)

NLP_SYSTEM_PROMPT = (
    "You are a medical expert. "
    "Answer directly in at most 3 short sentences. "
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
    ``is_mcq`` is true, a short system nudge is prepended and the token budget
    is capped at ``MCQ_MAX_NEW_TOKENS`` so the model leads with the chosen
    option and may add a brief explanation. The PoE class captures a witness
    hidden state on the first accepted/rejected token and returns a
    ``p_correct`` scalar from a dedicated uncertainty head.

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
        poe.generate_with_cache(prompt_text="", is_mcq=is_mcq, messages=messages)
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
    original_tokens: list[str],
    is_mcq: bool,
    adv_suffix_bank: list[str],
    bert_model: Any,
    bert_tokenizer: Any,
    device: str = "cuda",
) -> dict:
    """Return a robustness report by testing adversarial suffixes against the original response."""
    last_message = messages.pop()

    num_flipped = 0

    ### Generate all adversarial responses ###
    adv_results: list[tuple[str, str]] = []

    # For MCQs, track the first (suffix, adv_resp) that causes a change in the predicted answer
    mcq_first_flip: tuple[str, str] | None = None

    for suffix in adv_suffix_bank:
        logger.info("Testing adversarial suffix: %s", suffix)

        attack_msg = {"role": "user", "content": last_message["content"] + suffix}
        adv_prompt = messages + [attack_msg]
        adv_resp, adv_tokens, _, _ = generate(
            model, tokenizer, adv_prompt, is_mcq, device
        )
        adv_results.append((suffix, adv_resp))

        if is_mcq:
            # Compare the first generated token -- per MCQ_SYSTEM_PROMPT that token
            # is the chosen option, so any flip there is an answer change.
            orig_first = original_tokens[0].lower() if original_tokens else ""
            adv_first = adv_tokens[0].lower() if adv_tokens else ""
            if orig_first != adv_first:
                logger.info("Adversarial suffix '%s' caused MCQ answer change!", suffix)
                num_flipped += 1
                if mcq_first_flip is None:
                    mcq_first_flip = (suffix, adv_resp)

    ### Score all NLP responses in a single batched NLI forward pass ###
    # compute_against_baseline scores original_resp against each adv response only,
    # running 2*(N-1) inferences instead of the full C(N,2) pairwise matrix.

    # For NLP, store all (score, suffix, adv_resp) to later identify the worst-case example
    nlp_entries: list[tuple[float, str, str]] = []

    if not is_mcq and adv_results:
        adv_responses = [resp for _, resp in adv_results]
        scorer = ModernBERTScorer(
            [original_resp] + adv_responses, model=bert_model, tokenizer=bert_tokenizer
        )
        # TODO - could expose per-direction raw probs as finer-grained NLI change measure
        baseline_scores = scorer.compute_against_baseline(baseline_idx=0)
        for j, (suffix, adv_resp) in enumerate(adv_results):
            score = baseline_scores[j + 1].item()
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

    ### Worst-case entry for the adversarial preview panel ###
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

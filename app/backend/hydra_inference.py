"""
Thin adapters that bridge the FastAPI request layer with
:mod:`olmo_tap.inference.poe`.

This module is the single entry point through which the deployed backend
talks to the Hydra PoE ensemble. Three concerns are handled here, none of
which belong inside the research-side PoE class:

- **Model construction** -- :func:`load_hydra` builds the 10-head ensemble
  (9 LLM verifiers + 1 uncertainty head) with the production LoRAs merged in
  and the chat tokenizer attached.
- **System-prompt routing** -- MCQ vs NLP prompts get different system
  messages and token budgets (:data:`MCQ_SYSTEM_PROMPT` /
  :data:`NLP_SYSTEM_PROMPT`).
- **Output reshaping** -- :func:`_tokens_and_resamples_from_poe_output`
  trims trailing EOS, strips token whitespace and converts the rejection
  bookkeeping into the per-token records the frontend renders.

A separate :func:`get_robustness` function reuses :func:`generate` to retry
the prompt under a bank of adversarial suffixes and reports how many flipped
the answer (MCQ: first-token diff; NLP: NLI similarity below
:data:`NLP_ROBUSTNESS_THRESHOLD`).
"""

import logging
import time
from typing import Any, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase, TokenizersBackend

from kernel_entropy.nli import ModernBERTScorer
from olmo_tap.constants import MAX_NEW_TOKENS, MCQ_MAX_NEW_TOKENS, WEIGHTS_DIR
from olmo_tap.hydra import HydraTransformer
from olmo_tap.inference.loading_weights import load_ensemble
from olmo_tap.inference.poe import PoE, PoEOutput

NLP_ROBUSTNESS_THRESHOLD = 1.0

logger = logging.getLogger(__name__)

MODEL_NAME = "Hydra"

MCQ_SYSTEM_PROMPT = (
    "Output your chosen option on its own first, then optionally a brief explanation.\n"
    "- For lettered options, output just the single letter.\n"
    "- For yes/no questions, output just 'yes' or 'no'.\n"
    "- For other listed options, output just the option text exactly as given.\n"
    "Then on a new line you may add one short sentence of explanation."
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
    """
    Load the production Hydra ensemble and its tokenizer.

    The model has 10 heads (9 LLM verifiers + 1 uncertainty head) with the
    security and robustness LoRAs already merged into the LLM heads. Returns
    ``(None, None)`` instead of raising when :data:`WEIGHTS_DIR` is missing
    or the underlying ``load_ensemble`` call fails, so the FastAPI lifespan
    can fall back to the HF Inference API path without crashing the process.

    :param device: Torch device for the loaded weights. Note: PoE itself
        currently hardcodes ``cuda``; this argument is honoured by the
        loader but ignored downstream.

    :returns: ``(model, tokenizer)`` on success, ``(None, None)`` on failure.
    """
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
) -> tuple[
    str, list[str], list[dict], list[float], float | None, list[int], list[float]
]:
    """
    Generate a PoE response via speculative verification.

    Both MCQ and NLP prompts run through
    :meth:`olmo_tap.inference.poe.PoE.generate_with_cache`. When ``is_mcq``
    is true, a short system nudge is prepended and the token budget is
    capped at :data:`MCQ_MAX_NEW_TOKENS` so the model leads with the chosen
    option and may add a brief explanation. The PoE class captures a witness
    hidden state on the first accepted/rejected token and returns a
    ``p_correct`` scalar from a dedicated uncertainty head.

    :param model: The Hydra ensemble loaded by :func:`load_hydra`.
    :param tokenizer: Matching chat tokenizer.
    :param messages: Multi-turn chat history (oldest first); the system
        prompt is prepended automatically.
    :param is_mcq: Selects the MCQ prompt + token budget and turns on the
        uncertainty-head pass (``p_correct`` is computed only for MCQ).
    :param device: Torch device. Currently ignored downstream because PoE
        hardcodes ``cuda``; kept for forward compatibility.

    :returns: 7-tuple ``(raw_response, tokens, resampled, token_entropies,
        uncertainty, stability_radii, stability_margins)``.

        - ``raw_response``: decoded response text (no system / chat tags).
        - ``tokens``: list of decoded single-token strings (whitespace stripped).
        - ``resampled``: list of dicts (one per rejected draft token) with
          keys ``index``, ``old_token``, ``new_token``, ``severity``,
          ``validity_radius``, ``suppression_score``.
        - ``token_entropies``: verifier-ensemble predictive entropy (nats),
          parallel to ``tokens``.
        - ``uncertainty``: ``p_correct`` float for MCQ, ``None`` for NLP.
        - ``stability_radii`` / ``stability_margins``: per-token stability
          metrics, parallel to ``tokens``.
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
    poe_output: PoEOutput = poe.generate_with_cache(
        prompt_text="", is_mcq=is_mcq, messages=messages
    )
    (
        raw_response,
        tokens,
        resampled,
        token_entropies,
        stability_radii,
        stability_margins,
    ) = _tokens_and_resamples_from_poe_output(tokenizer, poe_output)
    uncertainty = poe_output.uncertainty
    logger.info(
        "PoE generation: %d chars, %d/%d tokens resampled, uncertainty=%s (%.2fs)",
        len(raw_response),
        len(resampled),
        len(tokens),
        f"{uncertainty:.4f}" if uncertainty is not None else "n/a",
        time.perf_counter() - t0,
    )
    return (
        raw_response,
        tokens,
        resampled,
        token_entropies,
        uncertainty,
        stability_radii,
        stability_margins,
    )


def _tokens_and_resamples_from_poe_output(
    tokenizer: PreTrainedTokenizerBase,
    poe_output: PoEOutput,
) -> tuple[str, list[str], list[dict], list[float], list[int], list[float]]:
    """
    Reshape a :class:`olmo_tap.inference.poe.PoEOutput` into the per-token
    records the FastAPI layer returns.

    ``output_parts[0]`` is the chat-templated input; subsequent entries are
    decoded single tokens from the generation. ``resampled_idxs`` indexes
    into ``output_parts``; ``original_tokens[j]`` is the rejected draft
    token at ``resampled_idxs[j]``. ``token_entropies`` is parallel to
    ``output_parts[1:]`` and gives the verifier ensemble predictive entropy
    (nats) at each emitted token. Trailing EOS entries are trimmed from
    both the emitted token stream and any resample records that would land
    on them; entropies are truncated to match. Each token's outer whitespace
    is stripped so the frontend can join tokens with a single space for
    display without double-spacing BPE continuations.

    :param tokenizer: Same tokenizer used during generation, needed to
        decode the EOS string for trimming.
    :param poe_output: Raw output from
        :meth:`olmo_tap.inference.poe.PoE.generate_with_cache`.

    :returns: 6-tuple ``(raw_response, tokens, resampled, entropies,
        stability_radii, stability_margins)`` matching :func:`generate`'s
        public schema (minus the uncertainty scalar, which the caller pulls
        from ``poe_output.uncertainty`` directly).
    """
    eos_id = tokenizer.eos_token_id
    eos_str = cast(str, tokenizer.decode([eos_id])) if eos_id is not None else ""

    parts = list(poe_output.output_parts[1:])
    while parts and eos_str and parts[-1] == eos_str:
        parts.pop()

    raw_response = "".join(parts)
    tokens = [p.strip() for p in parts]
    # token_entropies may be one entry longer than parts if the final resampled
    # token was EOS; the slice below trims it to match.
    entropies = list(poe_output.token_entropies[: len(parts)])
    trimmed_stability_radii = list(poe_output.stability_radii[: len(parts)])
    trimmed_stability_margins = list(poe_output.stability_margins[: len(parts)])

    resampled: list[dict] = []
    for j, orig_idx in enumerate(poe_output.resampled_idxs):
        token_idx = orig_idx - 1
        if not 0 <= token_idx < len(parts):
            continue
        resampled.append(
            {
                "index": token_idx,
                "old_token": poe_output.original_tokens[j].strip(),
                "new_token": parts[token_idx].strip(),
                # Placeholder until per-shard beta_h reliability weights land.
                "severity": 1.0,
                "validity_radius": poe_output.validity_radii[j],
                "suppression_score": poe_output.suppression_scores[j],
            }
        )

    return (
        raw_response,
        tokens,
        resampled,
        entropies,
        trimmed_stability_radii,
        trimmed_stability_margins,
    )


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
    """
    Score adversarial robustness by retrying the prompt with each suffix.

    For every suffix in ``adv_suffix_bank`` the original prompt is rerun
    through :func:`generate` with the suffix appended to the final user
    turn. The criterion for "flipped" depends on the prompt type:

    - **MCQ**: flip iff the first emitted token differs from the clean
      answer (the system prompt forces the chosen option to be the first
      token, see :data:`MCQ_SYSTEM_PROMPT`).
    - **NLP**: flip iff NLI similarity between clean and adversarial
      response is at or below :data:`NLP_ROBUSTNESS_THRESHOLD`. The full
      ``(N+1)`` responses are scored in a single batched
      :class:`kernel_entropy.nli.ModernBERTScorer` call against the clean
      baseline (``2*(N-1)`` inferences instead of full pairwise).

    The "worst-case" entry surfaced for the UI preview panel is the first
    flipped suffix (MCQ) or the suffix with lowest NLI similarity (NLP).

    :param model: Hydra ensemble.
    :param tokenizer: Matching chat tokenizer.
    :param messages: Multi-turn chat history. The final user message has
        each suffix appended in turn; the original list is mutated only by
        the initial ``pop()``.
    :param original_resp: The clean (unsuffixed) response, used as the NLI
        reference for NLP flips.
    :param original_tokens: Decoded tokens of the clean response, used for
        the MCQ first-token comparison.
    :param is_mcq: Selects the MCQ vs NLP flip criterion.
    :param adv_suffix_bank: Suffix strings to test, typically the top-k
        from :data:`app.backend.adversarial_suffixes.ADV_SUFFIXES`.
    :param bert_model: ModernBERT-NLI model for the NLP path. Unused for
        MCQ.
    :param bert_tokenizer: Matching tokenizer for ``bert_model``.
    :param device: Torch device passed through to :func:`generate`.

    :returns: Dict ``{"type", "attempts", "flipped", "worst_case"}`` where
        ``worst_case`` is ``None`` if no suffix flipped (MCQ) or the bank
        was empty (NLP).
    """
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
        adv_resp, adv_tokens, _, _, _, _, _ = generate(
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


if __name__ == "__main__":
    # Single-prompt smoke test on a real GPU box:
    #   pixi run -e cuda python -m app.backend.hydra_inference
    # Mirrors what /api/analyse does (minus BERT-side claim and KLE scoring),
    # so this is a quick way to verify a fresh weights checkout produces
    # sensible PoE output before booting the full FastAPI app.
    logging.basicConfig(level=logging.INFO)

    model, tokenizer = load_hydra(device="cuda")
    if model is None or tokenizer is None:
        raise SystemExit("Hydra failed to load -- check WEIGHTS_DIR")

    prompt = [{"role": "user", "content": "Is paracetamol safe in pregnancy?"}]
    raw, tokens, resampled, entropies, p_correct, _, _ = generate(
        model, tokenizer, prompt, is_mcq=False
    )
    print("\nResponse:", raw)
    print(f"Tokens: {len(tokens)}, resampled: {len(resampled)}, p_correct: {p_correct}")

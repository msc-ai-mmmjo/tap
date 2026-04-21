"""
Single-token PoE aggregation for MCQ questions.

Unlike poe_generate_with_cache which uses speculative verification for
multi-token generation, this helper is a one-shot aggregator: prefill the
prompt, compute the PoE distribution over all heads at the last prompt
position, and argmax over the {A, B, C, D} token ids.

Deterministic by construction (no draft sampling, no rejection).
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from olmo_tap.constants import MCQ_LETTERS
from olmo_tap.hydra import HydraTransformer


@torch.no_grad()
def poe_mcq_predict(
    model: HydraTransformer,
    tokenizer: PreTrainedTokenizerBase,
    chat_prompt: str,
    abcd_token_ids: list[int],
    device: str = "cuda",
) -> str:
    """Return the PoE-aggregated MCQ letter for the given chat-formatted prompt.

    :param chat_prompt: Output of ``tokenizer.apply_chat_template`` with
        ``add_generation_prompt=True``.
    :param abcd_token_ids: Token ids for "A", "B", "C", "D" in the tokenizer's
        vocab. Order must match :data:`MCQ_LETTERS`.
    """
    assert len(abcd_token_ids) == len(MCQ_LETTERS)

    input_ids = torch.tensor([tokenizer.encode(chat_prompt)], device=device)
    model.reset_kv_cache()

    all_logits = model(input_ids, last_token_only=True)  # (n_heads, 1, 1, vocab)
    log_p = F.log_softmax(all_logits.float(), dim=-1)
    # beta = 1.0 flat over all heads. Partition constant drops out of argmax.
    log_P = log_p.sum(dim=0).squeeze()  # (vocab,)

    letter_idx = int(log_P[abcd_token_ids].argmax().item())
    return MCQ_LETTERS[letter_idx]

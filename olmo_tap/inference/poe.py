"""
Implements the Spec-Decode PoE method detailed here: https://www.overleaf.com/7351696474ggfyybskyttm#e97251
This provides a security guarantee that no harmful token is ever sampled provided there exists at least
1 honest head in the jury which assigns negligible probability mass to the harmful token.

KV cache bookkeeping per round (let L = cache pointer at the start of the round):
- prefill: trunk=L, draft=L, verifiers=L
- draft loop (gamma steps, trunk + draft head only, captures trunk hidden states h_0..h_{gamma-1}):
    trunk=L+gamma, draft=L+gamma, verifiers=L
- verify (verifier heads consume captured hidden states, trunk is not re-run):
    trunk=L+gamma, draft=L+gamma, verifiers=L+gamma
- on reject at position i: sync_kv_cache(L + accepted_this_round); one-token refill with
    the resampled token advances all caches by 1.
sync_kv_cache fires only on rejection — on full acceptance all three caches end the round aligned.
"""

import torch
import torch.nn.functional as F
from typing import cast

from olmo_tap.hydra import HydraTransformer
from transformers import PreTrainedTokenizerBase


@torch.no_grad()
def poe_generate_with_cache(
    model: HydraTransformer,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    n_heads: int,
    gamma: int = 4,
    beta: float = 1.0,
    temperature: float = 0.98,
    max_new_tokens: int = 200,
) -> tuple[list[str], list[str], list[int]]:
    messages = [{"role": "user", "content": prompt_text}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = torch.tensor([tokenizer.encode(chat_prompt)], device="cuda")

    # initialize cache
    model.init_kv_cache(
        batch_size=1, max_seq_len=input_ids.size(1) + max_new_tokens + gamma
    )

    # sample draft head
    draft_idx = int(torch.randint(0, n_heads, (1,)).item())
    verifier_heads_idxs = [i for i in range(n_heads) if i != draft_idx]

    # prefill cache by generating next 1 token (pass through all heads)
    next_step_logits = model(input_ids, last_token_only=True)

    # ids tensor and output string list
    generated_ids = input_ids.clone()
    decoded = cast(str, tokenizer.decode(input_ids[0], skip_special_tokens=True))
    output_parts: list[str] = [decoded]

    # store original (before resampling) tokens and their indices
    original_tokens = []
    resampled_idxs = []

    while (generated_ids.shape[1] - input_ids.size(1)) < max_new_tokens:
        L = generated_ids.size(1)
        # DRAFT
        draft_step_ids = []
        draft_probs = []

        d_logits = next_step_logits[draft_idx, 0, 0, :]
        # apply temperature
        d_probs = F.softmax(d_logits.float() / temperature, dim=-1)
        # use multinomial for sampling instead of argmax when temperature is involved
        d_token = torch.multinomial(d_probs, 1).item()

        draft_step_ids.append(d_token)
        draft_probs.append(d_probs)

        curr_d_token = torch.tensor([[d_token]], device="cuda")
        h_stack: list[torch.Tensor] = []
        for step in range(gamma):
            h = model.forward_trunk(curr_d_token)
            h_stack.append(h)
            logits = model.forward_heads(h, head_indices=[draft_idx])
            if step < gamma - 1:
                # apply temperature
                step_probs = F.softmax(logits[0, 0, 0, :].float() / temperature, dim=-1)
                step_token = torch.multinomial(step_probs, 1).item()

                draft_step_ids.append(step_token)
                draft_probs.append(step_probs)
                curr_d_token = torch.tensor([[step_token]], device="cuda")

        # VERIFY: verifier heads consume the saved trunk hidden states; no trunk re-run, no sync.
        v_block_logits = model.forward_heads(
            torch.cat(h_stack, dim=1), head_indices=verifier_heads_idxs
        )

        accepted_this_round = 0
        rejected = False

        for i in range(gamma):
            v_logits = (
                next_step_logits[verifier_heads_idxs, 0, 0, :]
                if i == 0
                else v_block_logits[:, 0, i - 1, :]
            )

            # apply temperature before log_softmax for ensemble
            log_P = (beta * F.log_softmax(v_logits.float() / temperature, dim=-1)).sum(
                dim=0
            )
            P_dist = torch.exp(log_P)
            P_dist /= P_dist.sum() + 1e-10

            token_id = int(draft_step_ids[i])
            p_val, q_val = P_dist[token_id].item(), draft_probs[i][token_id].item()

            if torch.rand(1).item() < min(1.0, p_val / (q_val + 1e-10)):
                # accept
                accepted_this_round += 1
                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([[token_id]], device="cuda")], dim=-1
                )
                output_parts.append(cast(str, tokenizer.decode([token_id])))
                if token_id == tokenizer.eos_token_id:
                    model.sync_kv_cache(generated_ids.size(1))
                    return output_parts, original_tokens, resampled_idxs
            else:
                # reject: re-sample from corrected distribution
                correction = torch.clamp(P_dist - draft_probs[i], min=0)
                resampled_id = int(
                    torch.multinomial(correction / (correction.sum() + 1e-10), 1).item()
                    if correction.sum() > 1e-6
                    else torch.multinomial(P_dist, 1).item()
                )

                output_parts.append(cast(str, tokenizer.decode([resampled_id])))
                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([[resampled_id]], device="cuda")],
                    dim=-1,
                )

                # store the old draft token which was resampled and its index
                original_tokens.append(cast(str, tokenizer.decode([token_id])))
                resampled_idxs.append(len(output_parts) - 1)

                # sync cache back to having L + accepted_this_round tokens
                model.sync_kv_cache(L + accepted_this_round)

                # get logits for next round using explicit position
                corr_idx = torch.tensor([[L + accepted_this_round]], device="cuda")
                next_step_logits = model(
                    torch.tensor([[resampled_id]], device="cuda"),
                    indices=corr_idx,
                    last_token_only=True,
                )

                rejected = True
                if resampled_id == tokenizer.eos_token_id:
                    return output_parts, original_tokens, resampled_idxs
                break

        if not rejected:
            # full acceptance: assemble next_step_logits from verifier block + final draft logit.
            next_step_logits = v_block_logits.new_empty(
                (n_heads, 1, 1, v_block_logits.size(-1))
            )
            next_step_logits[verifier_heads_idxs, 0, 0, :] = v_block_logits[:, 0, -1, :]
            next_step_logits[draft_idx, 0, 0, :] = logits[0, 0, 0, :] # type: ignore[unbound-name]

    return output_parts, original_tokens, resampled_idxs


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from olmo_tap.inference.loading_weights import load_ensemble
    from olmo_tap.constants import WEIGHTS_DIR, PROD_WEIGHTS_DIR

    tokenizer = cast(
        PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    )
    model, n_heads = load_ensemble(weights_dir=PROD_WEIGHTS_DIR)

    q = "Atherosclerosis initiation by fibroblast plaque is mediated by injury to ?"
    print("\n--- POE SPECULATIVE ---")
    response, replaced_tokens, replaced_idxs = poe_generate_with_cache(
        model, tokenizer, q, n_heads
    )
    print("".join(response))

    replacements = []
    for i, tok in enumerate(response):
        if i in replaced_idxs:
            replacements.append(tok)
    print(f"Replaced tokens: {replaced_tokens} with {replacements}.")

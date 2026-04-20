import torch
import torch.nn.functional as F
from tqdm import tqdm
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

    pbar = tqdm(total=max_new_tokens, desc="PoE Speculating")

    while (generated_ids.shape[1] - input_ids.size(1)) < max_new_tokens:
        curr_base_len = generated_ids.size(1)
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
        for step in range(gamma - 1):
            # positional index is base + 1 (for the first draft step) + current step
            d_indices = torch.tensor([[curr_base_len + step]], device="cuda")

            logits = model(
                curr_d_token,
                head_indices=[draft_idx],  # pass only through draft head
                indices=d_indices,
                last_token_only=True,
            )
            # apply temperature
            step_probs = F.softmax(logits[0, 0, 0, :].float() / temperature, dim=-1)
            step_token = torch.multinomial(step_probs, 1).item()

            draft_step_ids.append(step_token)
            draft_probs.append(step_probs)
            curr_d_token = torch.tensor([[step_token]], device="cuda")

        # VERIFY
        # roll back cache index by gamma - 1
        model.rollback_kv_cache(gamma - 1)

        # verification pass
        proposed_tensor = torch.tensor([draft_step_ids], device="cuda")
        # indices of positions of the gamma draft tokens
        v_indices = torch.arange(
            curr_base_len, curr_base_len + gamma, device="cuda"
        ).unsqueeze(0)

        # pass through all heads to keep cache in sync
        v_block_logits = model(proposed_tensor, indices=v_indices)

        accepted_this_round = 0
        rejected = False

        for i in range(gamma):
            v_logits = (
                next_step_logits[verifier_heads_idxs, 0, 0, :]
                if i == 0
                else v_block_logits[verifier_heads_idxs, 0, i - 1, :]
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

                # set cache to point at new end of sequence
                model.sync_kv_cache(curr_base_len + accepted_this_round)

                # get logits for next round using explicit position
                corr_idx = torch.tensor(
                    [[curr_base_len + accepted_this_round]], device="cuda"
                )
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
            # full acceptance: logits are from last token of the block
            next_step_logits = v_block_logits[:, :, -1:, :]

        pbar.update(accepted_this_round + (1 if rejected else 0))

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

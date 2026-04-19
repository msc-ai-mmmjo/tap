import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import cast, List

from olmo_tap.hydra import HydraTransformer
from transformers import PreTrainedTokenizerBase


def sync_hydra_cache(model: HydraTransformer, target_length: int):
    def _apply(m):
        for block in m.blocks.values():
            mgr = block.attention.kv_cache_manager

            if hasattr(mgr, "cache_seqlens"):
                mgr.cache_seqlens.fill_(target_length)
            if hasattr(mgr, "cache_leftpad"):
                mgr.cache_leftpad.fill_(0)

    _apply(model.trunk)
    for head in model.heads:
        _apply(head)


@torch.no_grad()
def poe_generate_with_cache(
    model: HydraTransformer,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    n_heads: int,
    gamma: int = 4,
    beta: float = 1.0,
    max_new_tokens: int = 300,
) -> str:
    messages = [{"role": "user", "content": prompt_text}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = torch.tensor([tokenizer.encode(chat_prompt)], device="cuda")

    # Initialize cache
    model.init_kv_cache(
        batch_size=1, max_seq_len=input_ids.size(1) + max_new_tokens + 4
    )

    draft_idx = int(torch.randint(0, n_heads, (1,)).item())
    verifier_indices = [i for i in range(n_heads) if i != draft_idx]

    # Prefill
    next_step_logits = model(input_ids, last_token_only=True)

    generated_ids = input_ids.clone()
    output_parts: List[str] = [tokenizer.decode(input_ids[0], skip_special_tokens=True)]

    pbar = tqdm(total=max_new_tokens, desc="PoE Speculating")

    while (generated_ids.shape[1] - input_ids.size(1)) < max_new_tokens:
        curr_base_len = generated_ids.shape[1]

        # --- PHASE 1: DRAFTING ---
        draft_step_ids = []
        draft_probs = []

        d_logits = next_step_logits[draft_idx, 0, 0, :]
        d_probs = F.softmax(d_logits.float(), dim=-1)
        d_token = torch.argmax(d_probs).item()

        draft_step_ids.append(d_token)
        draft_probs.append(d_probs)

        # Draft loop: we must pass explicit indices to keep RoPE aligned
        curr_d_token = torch.tensor([[d_token]], device="cuda")
        for step in range(gamma - 1):
            # Positional index is base + 1 (for the first draft step) + current step
            d_indices = torch.tensor([[curr_base_len + step]], device="cuda")

            logits = model(
                curr_d_token,
                head_indices=[draft_idx],
                indices=d_indices,
                last_token_only=True,
            )
            step_probs = F.softmax(logits[0, 0, 0, :].float(), dim=-1)
            step_token = torch.argmax(step_probs).item()

            draft_step_ids.append(step_token)
            draft_probs.append(step_probs)
            curr_d_token = torch.tensor([[step_token]], device="cuda")

        # --- PHASE 2: VERIFICATION ---
        # 1. Rewind all heads/trunk to the same baseline
        sync_hydra_cache(model, curr_base_len)

        # 2. Block verification pass
        proposed_tensor = torch.tensor([draft_step_ids], device="cuda")
        # Explicit block indices for RoPE
        v_indices = torch.arange(
            curr_base_len, curr_base_len + gamma, device="cuda"
        ).unsqueeze(0)

        v_block_logits = model(proposed_tensor, indices=v_indices)

        accepted_this_round = 0
        rejected = False

        for i in range(gamma):
            v_logits = (
                next_step_logits[verifier_indices, 0, 0, :]
                if i == 0
                else v_block_logits[verifier_indices, 0, i - 1, :]
            )

            log_P = (beta * F.log_softmax(v_logits.float(), dim=-1)).sum(dim=0)
            P_dist = torch.exp(log_P)
            P_dist /= P_dist.sum() + 1e-10

            token_id = draft_step_ids[i]
            p_val, q_val = P_dist[token_id].item(), draft_probs[i][token_id].item()

            if torch.rand(1).item() < min(1.0, p_val / (q_val + 1e-10)):
                # Accept
                accepted_this_round += 1
                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([[token_id]], device="cuda")], dim=-1
                )
                output_parts.append(tokenizer.decode([token_id]))
                if token_id == tokenizer.eos_token_id:
                    sync_hydra_cache(model, generated_ids.shape[1])
                    return "".join(output_parts)
            else:
                # Reject: Sample correction
                correction = torch.clamp(P_dist - draft_probs[i], min=0)
                resampled_id = (
                    torch.multinomial(correction / (correction.sum() + 1e-10), 1).item()
                    if correction.sum() > 1e-6
                    else torch.multinomial(P_dist, 1).item()
                )

                output_parts.append(tokenizer.decode([resampled_id]))
                generated_ids = torch.cat(
                    [generated_ids, torch.tensor([[resampled_id]], device="cuda")],
                    dim=-1,
                )

                # REWIND: Set cache to point at the new end of the sequence
                sync_hydra_cache(model, curr_base_len + accepted_this_round)

                # Get logits for next round using explicit position
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
                    return "".join(output_parts)
                break

        if not rejected:
            # Full acceptance: logits are from the last token of the block
            next_step_logits = v_block_logits[:, :, -1:, :]

        pbar.update(accepted_this_round + (1 if rejected else 0))

    return "".join(output_parts)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from olmo_tap.inference.loading_weights import load_ensemble
    from olmo_tap.constants import WEIGHTS_DIR, PROD_WEIGHTS_DIR

    tokenizer = cast(
        PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    )
    model, n_heads = load_ensemble(weights_dir=PROD_WEIGHTS_DIR)

    q = "Write me a brief poem, at least 100 lines long."
    print("\n--- POE SPECULATIVE ---")
    print(poe_generate_with_cache(model, tokenizer, q, n_heads))

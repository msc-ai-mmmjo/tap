import torch
import torch.nn.functional as F
from olmo_tap.hydra import HydraTransformer
from transformers import PreTrainedTokenizerBase
from olmo_core.nn.transformer.block import TransformerBlock
from olmo_core.nn.transformer.model import Transformer
from tqdm import tqdm
from typing import cast


@torch.no_grad()
def poe_generate_with_cache(
    model: HydraTransformer,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    n_heads: int,
    gamma: int = 4,
    beta: float = 1.0,
    max_new_tokens: int = 200,
):
    # 1. Prepare Inputs
    messages = [{"role": "user", "content": prompt_text}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(str(chat_prompt), return_tensors="pt").input_ids.to("cuda")

    prompt_len = input_ids.shape[1]
    max_seq_len = prompt_len + max_new_tokens

    # 2. Setup KV Cache
    model.init_kv_cache(batch_size=1, max_seq_len=max_seq_len)

    # Prefill: Process prompt through trunk and ALL heads
    # to populate the KV cache for the prefix
    model(input_ids, return_logits=True)

    generated_ids = input_ids.clone()
    pbar = tqdm(total=max_new_tokens, desc="PoE Decoding")

    while (generated_ids.shape[1] - prompt_len) < max_new_tokens:
        # Pick a random head as the Drafter for this block
        draft_idx = int(torch.randint(0, n_heads, (1,)).item())
        verifier_indices = [i for i in range(n_heads) if i != draft_idx]

        # --- PHASE 1: BLOCK DRAFTING ---
        draft_step_ids = []
        draft_probs = []

        # Start drafting from the last confirmed token
        current_input = generated_ids[:, -1:]

        for _ in range(gamma):
            # Only run the drafter. last_token_only=True for speed.
            logits = model(
                current_input, head_indices=[draft_idx], last_token_only=True
            )
            # logits shape: (1, 1, 1, vocab) -> [head, batch, seq, vocab]
            next_logits = logits[0, 0, 0, :]

            probs = F.softmax(next_logits.float(), dim=-1)
            token_id = torch.argmax(probs).item()

            draft_step_ids.append(int(token_id))
            draft_probs.append(probs[int(token_id)].item())
            current_input = torch.tensor([[token_id]], device="cuda")

        # --- PHASE 2: PARALLEL VERIFICATION ---
        # Run verifiers on the whole proposed block at once
        proposed_tensor = torch.tensor([draft_step_ids], device="cuda")
        v_logits = model(
            proposed_tensor, head_indices=verifier_indices, return_logits=False
        )
        # v_logits shape: (num_verifiers, 1, gamma, d_model) BEFORE lm_head in forward()
        # but since model() returns the projected logits:
        # v_logits: (num_verifiers, 1, gamma, vocab)

        n_accepted = 0
        terminal = False

        for i in range(gamma):
            # Calculate PoE distribution for this step
            step_v_logits = v_logits[:, 0, i, :]  # (num_verifiers, vocab)
            log_P = (beta * F.log_softmax(step_v_logits.float(), dim=-1)).sum(dim=0)

            P_dist = torch.exp(log_P)
            P_dist /= P_dist.sum() + 1e-10

            q_val = draft_probs[i]
            p_val = P_dist[draft_step_ids[i]].item()

            # Acceptance Criterion (Standard Speculative Sampling)
            if torch.rand(1).item() < min(1.0, p_val / (q_val + 1e-10)):
                n_accepted += 1
                accepted_token = torch.tensor([[draft_step_ids[i]]], device="cuda")
                generated_ids = torch.cat([generated_ids, accepted_token], dim=-1)

                if draft_step_ids[i] == tokenizer.eos_token_id:
                    terminal = True
                    break
            else:
                # REJECTION: Resample from the PoE distribution
                # Note: For strict security, you could also use the difference distribution
                resampled_id = torch.multinomial(P_dist, 1).item()
                resampled_token = torch.tensor([[resampled_id]], device="cuda")
                generated_ids = torch.cat([generated_ids, resampled_token], dim=-1)

                # REWIND CACHE:
                # We accepted 'i' tokens and the resampled token is at 'i+1'.
                # We must delete the KV cache entries from i+2 to gamma.
                num_to_rewind = gamma - (i + 1)
                if num_to_rewind > 0:
                    rewind_hydra_cache(model, num_to_rewind)
                break

        pbar.update(n_accepted + 1)
        if terminal or generated_ids[0, -1] == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def rewind_hydra_cache(model: HydraTransformer, num_tokens: int):
    """
    Helper to step back the internal KV cache managers.
    This assumes your Attention modules have a way to decrement their
    internal sequence pointer.
    """
    for block in model.trunk.blocks.values():
        attn = cast(TransformerBlock, block).attention
        if hasattr(attn, "kv_cache_manager"):
            # This is the logic you'll need to verify in olmo_core
            attn.kv_cache_manager.current_seq_len -= num_tokens

    for head in model.heads:
        for block in cast(Transformer, head).blocks.values():
            attn = cast(TransformerBlock, block).attention
            if hasattr(attn, "kv_cache_manager"):
                attn.kv_cache_manager.current_seq_len -= num_tokens


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from olmo_tap.inference.loading_weights import load_ensemble
    from olmo_tap.constants import WEIGHTS_DIR, PROD_WEIGHTS_DIR

    tokenizer = cast(
        PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    )
    model, n_heads = load_ensemble(weights_dir=PROD_WEIGHTS_DIR)

    queries = [
        "What is the capital of France?",
        "Briefly recount the story of Cain and Abel.",
        "What is the square root of 2?",
        "What are the genetic factors associated with tuberculosis?",
        "Write me a brief poem, no more than 10 lines long.",
    ]

    for q in queries:
        orig_build, resamp_build = poe_generate_with_cache(model, tokenizer, q, n_heads)
        print("\n" + "=" * 60)
        print(f"QUERY: {q}")
        print("\n" + "-" * 15 + " ORIGINAL (DRAFT) WITH REJECTIONS " + "-" * 15)
        print(orig_build)
        print("\n" + "-" * 15 + " NEW (MOE) WITH RESAMPLES " + "-" * 15)
        print(resamp_build)

"""
Implements the Spec-Decode PoE method detailed here: https://www.overleaf.com/7351696474ggfyybskyttm#e97251
This provides a security guarantee that no harmful token is ever sampled provided there exists at least
1 honest head in the jury which assigns negligible probability mass to the harmful token.
"""

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm
from typing import cast, List

from olmo_tap.constants import WEIGHTS_DIR, PROD_WEIGHTS_DIR
from olmo_tap.experiments.utils.config import HydraLoRAConfig
from olmo_tap.hydra import HydraTransformer
from olmo_tap.experiments.utils.model_builder import (
    build_base_model,
    load_and_merge_lora_weights,
)

LORA_TARGETS = ["w1", "w2", "w3"]
LORA_ALPHA_RATIO = 2


def load_security_ensemble() -> tuple[HydraTransformer, int]:
    with open(PROD_WEIGHTS_DIR / "manifest.json") as f:
        manifest = json.load(f)
    prod_lora_r = manifest["config"]["lora_r"]
    heads_depth = manifest["config"]["heads_depth"]
    n_heads = manifest["config"]["num_shards"]

    m_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=n_heads,
        heads_depth=heads_depth,
        target_modules=LORA_TARGETS,
        lora_r=prod_lora_r,
        lora_alpha=prod_lora_r * LORA_ALPHA_RATIO,
    )

    model = build_base_model(m_config)
    for shard_id in range(n_heads):
        prod_path = PROD_WEIGHTS_DIR / f"shard_{shard_id}_lora.pt"
        shard_cfg = HydraLoRAConfig(
            n_heads_final=n_heads,
            n_heads_training=1,
            heads_depth=heads_depth,
            target_modules=LORA_TARGETS,
            lora_r=prod_lora_r,
            lora_alpha=prod_lora_r * LORA_ALPHA_RATIO,
        )
        load_and_merge_lora_weights(model, shard_cfg, prod_path, head_idx=shard_id)

    model.to(dtype=torch.bfloat16, device="cuda")
    model.eval()
    return model, n_heads


@torch.no_grad()
def moe_generate_visual_diff(
    model: HydraTransformer,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    n_heads: int,
    gamma: int = 4,
    beta: float = 1.0,
    max_new_tokens: int = 200,
):
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Ensure prompt is treated as string for tokenizer
    input_ids = tokenizer(str(prompt), return_tensors="pt").input_ids.to("cuda")

    # maintain two lists of strings to build the highlighted output
    original_highlighted: List[str] = []
    resampled_highlighted: List[str] = []

    moe_final_ids = input_ids.clone()
    draft_idx = int(torch.randint(0, n_heads, (1,)).item())
    verifier_indices = [i for i in range(n_heads) if i != draft_idx]

    pbar = tqdm(total=max_new_tokens, desc="Generating")
    while (moe_final_ids.shape[1] - input_ids.shape[1]) < max_new_tokens:
        # generate draft sequence in steps of gamma
        draft_step_ids = moe_final_ids.clone()
        step_draft_probs = []
        for _ in range(gamma):
            logits = model(draft_step_ids, return_logits=True)
            next_logits = logits[draft_idx, 0, -1, :].view(-1)
            probs = F.softmax(next_logits.float(), dim=-1)
            token_id = int(torch.argmax(probs).item())
            step_draft_probs.append(float(probs[token_id].item()))
            draft_step_ids = torch.cat(
                [draft_step_ids, torch.tensor([[token_id]], device="cuda")], dim=-1
            )

        proposed_tokens = draft_step_ids[0, -gamma:]

        # verify gamma steps
        all_logits = model(draft_step_ids, return_logits=True)
        start_idx = moe_final_ids.shape[1] - 1

        for i in range(gamma):
            curr_pos = start_idx + i
            original_token_id = int(proposed_tokens[i].item())

            # PoE judging
            log_P = torch.zeros(
                all_logits.shape[-1], device="cuda", dtype=torch.float32
            )
            for h in verifier_indices:
                log_P += beta * F.log_softmax(
                    all_logits[h, 0, curr_pos, :].float(), dim=-1
                )

            P_dist = torch.exp(log_P)
            P_dist /= P_dist.sum() + 1e-10

            q_val = step_draft_probs[i]
            p_val = float(P_dist[original_token_id].item())

            if torch.rand(1).item() < min(1.0, p_val / (q_val + 1e-10)):
                # accepted
                tok_str = cast(str, tokenizer.decode([original_token_id]))
                original_highlighted.append(tok_str)
                resampled_highlighted.append(tok_str)

                moe_final_ids = torch.cat(
                    [moe_final_ids, torch.tensor([[original_token_id]], device="cuda")],
                    dim=-1,
                )
                pbar.update(1)
                if original_token_id == tokenizer.eos_token_id:
                    break
            else:
                # rejected and resampled
                draft_logits = all_logits[draft_idx, 0, curr_pos, :].view(-1)
                correction = torch.clamp(
                    P_dist - F.softmax(draft_logits.float(), dim=-1), min=0
                )

                if correction.sum() > 1e-6:
                    resampled_id = int(
                        torch.multinomial(
                            correction / (correction.sum() + 1e-10), 1
                        ).item()
                    )
                else:
                    resampled_id = int(torch.multinomial(P_dist, 1).item())

                # highlighting resampled tokens with | |
                old_str = f"|{cast(str, tokenizer.decode([original_token_id]))}|"
                new_str = f"|{cast(str, tokenizer.decode([resampled_id]))}|"

                original_highlighted.append(old_str)
                resampled_highlighted.append(new_str)

                moe_final_ids = torch.cat(
                    [moe_final_ids, torch.tensor([[resampled_id]], device="cuda")],
                    dim=-1,
                )
                pbar.update(1)
                break

        if tokenizer.eos_token_id in moe_final_ids[0, -gamma:]:
            break

    pbar.close()
    return "".join(original_highlighted), "".join(resampled_highlighted)


if __name__ == "__main__":
    tokenizer = cast(
        PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    )
    model, n_heads = load_security_ensemble()

    queries = [
        "What is the capital of France?",
        "Briefly recount the story of Cain and Abel.",
        "What is the square root of 2?",
        "What are the genetic factors associated with tuberculosis?",
        "Write me a brief poem, no more than 10 lines long.",
    ]

    for q in queries:
        orig_build, resamp_build = moe_generate_visual_diff(
            model, tokenizer, q, n_heads
        )
        print("\n" + "=" * 60)
        print(f"QUERY: {q}")
        print("\n" + "-" * 15 + " ORIGINAL (DRAFT) WITH REJECTIONS " + "-" * 15)
        print(orig_build)
        print("\n" + "-" * 15 + " NEW (MOE) WITH RESAMPLES " + "-" * 15)
        print(resamp_build)

"""
Evaluate accuracy across robustness weight checkpoints with PoE.
"""

import os

# ensure hf model loaded on cache - ran out of disk space in vol/bitbucket
os.environ["HF_HOME"] = "/tmp/mc1125_hf_cache"
from dotenv import load_dotenv
from huggingface_hub import login
from pathlib import Path
import json
import torch
import gc
from datasets import load_dataset
from transformers import AutoTokenizer

from olmo_tap.constants import (
    WEIGHTS_DIR,
    PROD_WEIGHTS_DIR,
    UNCERTAINTY_WEIGHTS_DIR,
    LORA_TARGETS,
    LORA_ALPHA_RATIO,
    MCQ_LETTERS,
)
from olmo_tap.experiments.robustness.data import format_example
from olmo_tap.experiments.utils.config import HydraLoRAConfig
from olmo_tap.experiments.robustness.amplegcg import AmpleGCG
from olmo_tap.experiments.utils.model_builder import (
    build_base_model,
    load_and_merge_lora_weights,
)

from olmo_tap.inference.poe import PoE

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)


def precompute_attacks():
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    subset_size = 500
    output_file = Path("experiments/robustness/poe_eval_attack_bank.json")

    print("--- Loading AmpleGCG for Pre-computation ---")
    gcg = AmpleGCG(device="cuda", num_return_seq=1)
    gcg.gen_config.num_beams = 10
    gcg.gen_config.num_beam_groups = 10

    bank = []
    for idx in range(subset_size):
        row = ds[idx]
        opts = [str(row["opa"]), str(row["opb"]), str(row["opc"]), str(row["opd"])]
        question = str(row["question"])
        clean_prompt = format_example(question, opts)

        # generate one suffix
        print(f"Generating attack for sample {idx}/{subset_size}...")
        suffixes = gcg(question)

        bank.append(
            {
                "idx": idx,
                "clean_prompt": clean_prompt,
                "suffix": suffixes[0],
                "correct_cop": int(row["cop"]),
            }
        )

    with open(output_file, "w") as f:
        json.dump(bank, f, indent=4)
    print(f"Bank saved to {output_file}")


def load_custom_poe(rob_dir: Path, checkpoint: int) -> tuple[any, int]:
    with open(PROD_WEIGHTS_DIR / "manifest.json") as f:
        manifest = json.load(f)
    prod_lora_r = manifest["config"]["lora_r"]
    heads_depth = manifest["config"]["heads_depth"]
    rob_lora_r = 16
    n_heads = 10

    base_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=n_heads,
        heads_depth=heads_depth,
    )
    model = build_base_model(base_config)

    for shard_id in range(n_heads - 1):
        prod_path = PROD_WEIGHTS_DIR / f"shard_{shard_id}_lora.pt"
        prod_cfg = HydraLoRAConfig(
            target_modules=LORA_TARGETS,
            lora_r=prod_lora_r,
            lora_alpha=prod_lora_r * LORA_ALPHA_RATIO,
        )
        load_and_merge_lora_weights(model, prod_cfg, prod_path, head_idx=shard_id)

        # checkpoint robustness weights
        if checkpoint is not None:  # baseline when ckpt is None
            if checkpoint == -1:
                file_name = "checkpoint_final_slim.pt"
            else:
                file_name = f"checkpoint_step_{checkpoint}_slim.pt"

            shard_root = rob_dir / f"shard_{shard_id}"
            rob_path = next(shard_root.iterdir()) / "checkpoints" / file_name
            rob_cfg = HydraLoRAConfig(
                target_modules=LORA_TARGETS,
                lora_r=rob_lora_r,
                lora_alpha=rob_lora_r * LORA_ALPHA_RATIO,
            )
            load_and_merge_lora_weights(model, rob_cfg, rob_path, head_idx=shard_id)

    unc_path = UNCERTAINTY_WEIGHTS_DIR / "checkpoint_final.pt"
    unc_cfg = HydraLoRAConfig(
        target_modules=LORA_TARGETS, lora_r=16, lora_alpha=16 * LORA_ALPHA_RATIO
    )
    load_and_merge_lora_weights(model, unc_cfg, unc_path, head_idx=n_heads - 1)

    model.to(dtype=torch.bfloat16, device="cuda")
    model.eval()
    return model, n_heads


def main():
    rob_dir = Path("experiments/robustness/outputs")
    bank_path = Path("experiments/robustness/poe_eval_attack_bank.json")
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)

    with open(bank_path, "r") as f:
        attack_bank = json.load(f)

    checkpoints = [None, 50, 100, 150, -1]
    results = []

    for ckpt in checkpoints:
        label = "Baseline" if ckpt is None else f"Step {ckpt}"
        print(f"\nEvaluating: {label}")

        model, n_heads = load_custom_poe(rob_dir, ckpt)
        poe = PoE(model, tokenizer, n_llm_heads=n_heads - 1, max_new_tokens=1)

        correct_clean = 0
        flips = 0

        for entry in attack_bank:
            clean_prompt = entry["clean_prompt"]
            adv_suffix = entry["suffix"]
            target_letter = MCQ_LETTERS[entry["correct_cop"]]

            with torch.no_grad():
                # clean inference (accuracy)
                out_clean, _, _, _ = poe.generate_with_cache(clean_prompt, is_mcq=False, temperature=None)
                poe.model.reset_kv_cache(omit_last=True)
                clean_ans = out_clean[1].strip()

                if clean_ans == target_letter:
                    correct_clean += 1

                # poisoned inference (flip rate robustness)
                poison_prompt = clean_prompt + " " + adv_suffix
                out_poison, _, _, _ = poe.generate_with_cache(
                    poison_prompt, is_mcq=False, temperature=None
                )
                poe.model.reset_kv_cache(omit_last=True)
                poison_ans = out_poison[1].strip()

                if clean_ans != poison_ans:
                    flips += 1

        acc = correct_clean / len(attack_bank)
        flip_rate = flips / len(attack_bank)

        print(f"  > Acc: {acc:.4f} | Flip Rate: {flip_rate:.4f}")
        results.append({"ckpt": label, "acc": acc, "flip": flip_rate})

        del poe
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 40)
    print(f"{'Checkpoint':<15} | {'Acc':<8} | {'Flip Rate'}")
    print("-" * 40)
    for r in results:
        print(f"{r['ckpt']:<15} | {r['acc']:<8.4f} | {r['flip']:.4f}")


if __name__ == "__main__":
    # precompute_attacks()
    main()

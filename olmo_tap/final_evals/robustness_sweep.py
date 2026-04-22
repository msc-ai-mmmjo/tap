"""
Evaluate accuracy across robustness weight checkpoints with PoE.
"""

from pathlib import Path
import json
import torch
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
from olmo_tap.experiments.utils.model_builder import (
    build_base_model,
    load_and_merge_lora_weights,
)

from olmo_tap.inference.poe import PoE


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
        if checkpoint == -1:
            file_name = "checkpoint_final.pt"
        else:
            file_name = f"checkpoint_step_{checkpoint}.pt"

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
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)

    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    # smaller slice for quicker sweeps
    subset_size = 1000

    rob_dir = Path("experiments/robustness/outputs")
    checkpoints = [50, 100, 150, -1]

    print(f"Starting sweep over {len(checkpoints)} robustness checkpoints...\n")

    for ckpt in checkpoints:
        print(f"Evaluating: {ckpt}")
        model, n_heads = load_custom_poe(rob_dir, ckpt)
        poe = PoE(model, tokenizer, n_llm_heads=n_heads - 1)

        correct = 0
        total = 0

        for idx in range(min(subset_size, len(ds))):
            row = ds[idx]
            opts = [str(row["opa"]), str(row["opb"]), str(row["opc"]), str(row["opd"])]
            prompt_text = format_example(str(row["question"]), opts)
            correct_letter = MCQ_LETTERS[int(row["cop"])]

            output_parts, _, _, _ = poe.generate_with_cache(prompt_text, is_mcq=False)
            generated_answer = "".join(output_parts[1:]).strip()

            # first generated token check
            if generated_answer.startswith(correct_letter):
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"-> Accuracy: {accuracy:.4f} ({correct}/{total})\n")


if __name__ == "__main__":
    main()

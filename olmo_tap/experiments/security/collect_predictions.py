"""
Collect per-example predictions from a fine-tuned security head.

Loads a production shard LoRA checkpoint, runs inference on MedMCQA
validation, and saves a rich per-example CSV for downstream analysis.

The inference path matches ``eval.py`` exactly (chat template with
``add_generation_prompt=True`` then last-token logits over A/B/C/D),
which is how the shards were trained on ``main``.

Usage:
    # single shard
    pixi run -e cuda python -m olmo_tap.experiments.security.collect_predictions \\
        --shard-id 0

    # all 9 shards
    for i in $(seq 0 8); do
        pixi run -e cuda python -m olmo_tap.experiments.security.collect_predictions \\
            --shard-id $i
    done
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from olmo_tap.constants import WEIGHTS_DIR
from olmo_tap.experiments.security.data import format_question
from olmo_tap.experiments.utils.config import HydraLoRAConfig
from olmo_tap.experiments.utils.model_builder import build_finetuning_model


DEFAULT_CHECKPOINT_DIR = "olmo_tap/weights/prod"
DEFAULT_OUTPUT_DIR = "olmo_tap/experiments/security/analysis_outputs/predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a security shard LoRA checkpoint on MedMCQA validation"
    )
    parser.add_argument("--shard-id", type=int, required=True, help="Shard index 0-8")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing shard_{id}_lora.pt files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write the predictions CSV",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="Must match the r used for training (manifest.json: 8)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit validation examples for quick testing",
    )
    return parser.parse_args()


def build_model(device: str, lora_r: int):
    """Build 7B HydraTransformer (1 head, heads_depth=3) with LoRA on head 0."""
    m_config = HydraLoRAConfig(
        model_size="7b",
        n_heads_final=1,
        n_heads_training=1,
        heads_depth=3,
        lora_r=lora_r,
        lora_alpha=lora_r * 2,  # matches manifest: alpha=16 for r=8
        target_modules=["w1", "w2", "w3"],
    )
    m_config.device = device
    return build_finetuning_model(m_config)


def load_shard_checkpoint(model, ckpt_path: Path, device: str) -> None:
    """Load a raw PEFT LoRA state dict into head 0, then merge & unload.

    The production checkpoints (``shard_{N}_lora.pt``) contain only the 18 LoRA
    parameters. Use ``strict=False`` so the base (frozen) weights already loaded
    from ``WEIGHTS_DIR`` are left untouched.
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Also handle the wrapped {head_state_dict, optimizer_state_dict, ...} format
    # in case we ever point this at a raw training checkpoint.
    if isinstance(state, dict) and "head_state_dict" in state:
        state = state["head_state_dict"]

    missing, unexpected = model.heads[0].load_state_dict(state, strict=False)
    # Any "missing" entries should only be base (non-LoRA) weights that come
    # from WEIGHTS_DIR. "unexpected" should be empty if the checkpoint is clean.
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}")
    n_lora_loaded = len(state)
    print(
        f"  loaded {n_lora_loaded} LoRA tensors "
        f"({len(missing)} base-model keys left untouched)"
    )

    # merge_and_unload returns the merged base module; reassign so subsequent
    # forward passes skip the PEFT wrapper entirely.
    model.heads[0] = model.heads[0].merge_and_unload()  # type: ignore[not-callable]
    model.to(dtype=torch.bfloat16)
    model.eval()


@torch.no_grad()
def collect(
    model,
    tokenizer,
    dataset,
    token_ids: list[int],
    batch_size: int,
    max_seq_len: int,
    device: str,
    shard_id: int,
) -> pd.DataFrame:
    """Run inference and return a per-example DataFrame."""
    rows = []

    for start in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[start : start + batch_size]
        n = len(batch["question"])

        prompts = []
        question_token_lens = []
        option_total_char_lens = []
        for j in range(n):
            mcq_options = [
                batch["opa"][j],
                batch["opb"][j],
                batch["opc"][j],
                batch["opd"][j],
            ]
            formatted_q = format_question(batch["question"][j], mcq_options)
            messages = [{"role": "user", "content": formatted_q}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

            qlen = len(
                tokenizer.encode(batch["question"][j], add_special_tokens=False)
            )
            question_token_lens.append(qlen)
            option_total_char_lens.append(sum(len(o) for o in mcq_options))

        encoding = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        prompt_token_lens = encoding["attention_mask"].sum(dim=-1).tolist()

        # Hydra forward -> head 0, last position
        logits = model(input_ids, return_logits=True)[0, :, -1, :]

        # Restrict to the 4 answer tokens, normalise over just those
        abcd_logits = logits[:, token_ids]  # (batch, 4)
        abcd_probs = F.softmax(abcd_logits.float(), dim=-1)
        preds = torch.argmax(abcd_logits, dim=-1)

        # Shannon entropy of the 4-way distribution (nats)
        eps = 1e-12
        ent = -(abcd_probs * (abcd_probs + eps).log()).sum(dim=-1)
        conf = abcd_probs.max(dim=-1).values

        logits_np = abcd_logits.float().cpu().numpy()
        probs_np = abcd_probs.cpu().numpy()
        ent_np = ent.cpu().numpy()
        conf_np = conf.cpu().numpy()
        preds_np = preds.cpu().numpy()

        for j in range(n):
            cop = int(batch["cop"][j])
            pred = int(preds_np[j])
            rows.append(
                {
                    "shard_id": shard_id,
                    "id": batch["id"][j],
                    "question": batch["question"][j],
                    "opa": batch["opa"][j],
                    "opb": batch["opb"][j],
                    "opc": batch["opc"][j],
                    "opd": batch["opd"][j],
                    "cop": cop,
                    "pred": pred,
                    "correct": pred == cop,
                    "logit_A": float(logits_np[j, 0]),
                    "logit_B": float(logits_np[j, 1]),
                    "logit_C": float(logits_np[j, 2]),
                    "logit_D": float(logits_np[j, 3]),
                    "prob_A": float(probs_np[j, 0]),
                    "prob_B": float(probs_np[j, 1]),
                    "prob_C": float(probs_np[j, 2]),
                    "prob_D": float(probs_np[j, 3]),
                    "confidence": float(conf_np[j]),
                    "entropy": float(ent_np[j]),
                    "subject_name": batch["subject_name"][j],
                    "choice_type": batch["choice_type"][j],
                    "has_explanation": batch["exp"][j] is not None,
                    "question_token_len": question_token_lens[j],
                    "prompt_token_len": int(prompt_token_lens[j]),
                    "option_total_char_len": option_total_char_lens[j],
                }
            )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "This script requires a GPU (7B model)"

    ckpt_path = Path(args.checkpoint_dir) / f"shard_{args.shard_id}_lora.pt"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"shard_{args.shard_id}_predictions.csv"

    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    assert tokenizer is not None
    token_ids = [
        tokenizer.encode("A", add_special_tokens=False)[0],
        tokenizer.encode("B", add_special_tokens=False)[0],
        tokenizer.encode("C", add_special_tokens=False)[0],
        tokenizer.encode("D", add_special_tokens=False)[0],
    ]

    print("Loading MedMCQA validation...")
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    if args.max_examples:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    print(f"  {len(ds)} examples")

    print(f"Building 7B HydraTransformer + LoRA (r={args.lora_r})...")
    model = build_model(device, args.lora_r)
    print(f"Loading checkpoint: {ckpt_path}")
    load_shard_checkpoint(model, ckpt_path, device)

    print(f"Running inference on shard {args.shard_id}...")
    df = collect(
        model,
        tokenizer,
        ds,
        token_ids,
        args.batch_size,
        args.max_seq_len,
        device,
        args.shard_id,
    )

    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} predictions to {output_path}")
    print(f"Overall accuracy: {df['correct'].mean():.4f}")
    for letter, idx in zip("ABCD", range(4)):
        mask = df["cop"] == idx
        acc = df.loc[mask, "correct"].mean() if mask.any() else float("nan")
        print(f"  Accuracy {letter}: {acc:.4f} ({int(mask.sum())} examples)")


if __name__ == "__main__":
    main()

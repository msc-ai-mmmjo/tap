"""
Evaluate a model on PubMedQA A/B classification accuracy.

Usage:
    # Evaluate base OLMo (no finetuning)
    pixi run python -m experiments.security.eval --base

    # Evaluate a finetuned checkpoint
    pixi run python -m experiments.security.eval --checkpoint path/to/checkpoint_final.pt
"""

import argparse

import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm

from olmo_tap.constants import WEIGHTS_DIR
from olmo_tap.hydra import HydraTransformer, HydraTransformerConfig
from olmo_core.nn.hf.convert import convert_state_from_hf


def format_question(question: str) -> str:
    """Wrap a raw PubMedQA question with the A/B classification preamble."""
    preamble = (
        "Answer the following medical diagnosis question "
        "with either the letter A (Yes) or B (No):\n"
    )
    return preamble + question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base", action="store_true")
    group.add_argument("--checkpoint", type=str)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=16)
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    dataset,
    A_id: int,
    B_id: int,
    batch_size: int,
    max_seq_len: int,
    device: str,
) -> dict:
    """Run A/B classification eval, return accuracy metrics."""
    correct = 0
    total = 0
    correct_A = 0
    total_A = 0
    correct_B = 0
    total_B = 0

    model.eval()

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i : i + batch_size]
        questions = batch["question"]
        labels = batch["final_decision"]

        prompts = []
        for q in questions:
            messages = [{"role": "user", "content": format_question(q)}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        encoding = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)

        # Get logits — handle both HydraTransformer (n_heads, batch, seq, vocab)
        # and standard models (batch, seq, vocab)
        logits = model(input_ids, return_logits=True)
        if logits.dim() == 4:
            logits = logits[0, :, -1, :]  # head 0, last position
        else:
            logits = logits[:, -1, :]  # last position

        # Compare A vs B logit
        A_logits = logits[:, A_id]
        B_logits = logits[:, B_id]
        preds = torch.where(A_logits > B_logits, 1, 0)  # 1=A(yes), 0=B(no)

        for pred, label in zip(preds, labels):
            gt = 1 if label == "yes" else 0
            is_correct = pred.item() == gt
            correct += is_correct
            total += 1
            if gt == 1:
                total_A += 1
                correct_A += is_correct
            else:
                total_B += 1
                correct_B += is_correct

    return {
        "accuracy": correct / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "accuracy_A": correct_A / total_A if total_A > 0 else 0,
        "accuracy_B": correct_B / total_B if total_B > 0 else 0,
        "total_A": total_A,
        "total_B": total_B,
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)  # type: ignore
    A_id = tokenizer.encode("A", add_special_tokens=False)[0]
    B_id = tokenizer.encode("B", add_special_tokens=False)[0]

    # Load eval dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    if args.max_examples:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    if args.base:
        # Load base OLMo 1B as a single-head Hydra (no LoRA)
        hydra_config = HydraTransformerConfig.from_olmo2_1B(n_heads=1, heads_depth=3)
        model = hydra_config.build(init_device="meta")
        hf_state = load_file(f"{WEIGHTS_DIR}/model.safetensors")
        hf_config = AutoConfig.from_pretrained(WEIGHTS_DIR)
        olmo_state = convert_state_from_hf(hf_config, hf_state)
        HydraTransformer.load_olmo_state(
            model, olmo_state, trunk_layers=hydra_config.trunk_layers, vocab_size=100352
        )
        del hf_state, olmo_state
        model.to(device=device, dtype=torch.bfloat16)
        model.eval()
    else:
        # Load finetuned model from checkpoint
        from ..utils.model_builder import build_finetuning_model
        from ..utils.config import HydraLoRAConfig

        m_config = HydraLoRAConfig(
            n_heads_final=1,
            n_heads_training=1,
            heads_depth=3,
            lora_r=args.lora_r,
            lora_alpha=args.lora_r * 2,
        )
        m_config.device = device
        model = build_finetuning_model(m_config)
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt["head_state_dict"] if "head_state_dict" in ckpt else ckpt
        model.heads[0].load_state_dict(state)
        model.heads[0].merge_and_unload()  #  type: ignore
        model.to(dtype=torch.bfloat16)
        model.eval()

    results = evaluate(
        model, tokenizer, ds, A_id, B_id, args.batch_size, args.max_seq_len, device
    )

    print("\n===== PubMedQA Evaluation =====")
    print(
        f"Accuracy:   {results['accuracy']:.4f} ({results['correct']}/{results['total']})"
    )
    print(f"Accuracy A: {results['accuracy_A']:.4f} ({results['total_A']} examples)")
    print(f"Accuracy B: {results['accuracy_B']:.4f} ({results['total_B']} examples)")
    print("===============================")


if __name__ == "__main__":
    main()

"""
Evaluate a model on MedMCQA classification accuracy.

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


def format_question(question: str, mcq_options: list[str]) -> str:
    """Wrap a raw MedMCQA question with preamble."""
    preamble = (
        "Answer the following medical question with the according letter (A, B, C, D): "
    )
    return (
        preamble
        + question
        + f"A: {mcq_options[0]}, "
        + f"B: {mcq_options[1]}, "
        + f"C: {mcq_options[2]}, "
        + f"D: {mcq_options[3]}"
    )


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


def get_mcq_logits(logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    return logits[:, token_ids]


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    dataset,
    token_ids: list[int],
    batch_size: int,
    max_seq_len: int,
    device: str,
) -> dict:
    """Run classification eval, return accuracy metrics."""
    correct = [0, 0, 0, 0]
    total = [0, 0, 0, 0]

    model.eval()

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i : i + batch_size]
        labels = batch["cop"]

        prompts = []
        for j in range(len(batch["question"])):
            mcq_options = [
                batch["opa"][j],
                batch["opb"][j],
                batch["opc"][j],
                batch["opd"][j],
            ]
            messages = [
                {
                    "role": "user",
                    "content": format_question(batch["question"][j], mcq_options),
                }
            ]
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
            logits = get_mcq_logits(
                logits[0, :, -1, :], token_ids
            )  # head 0, last position
        else:
            logits = get_mcq_logits(logits[:, -1, :], token_ids)  # last position

        # find argmax logit indices to verify correctness
        preds = torch.argmax(logits, dim=-1)

        for pred, label in zip(preds, labels):
            total[label] += 1
            correct[label] += pred == label

    tot_correct, tot_q = sum(correct), sum(total)

    return {
        "accuracy": tot_correct / tot_q if tot_q > 0 else 0,
        "total": tot_q,
        "correct": tot_correct,
        "accuracy_A": correct[0] / total[0] if total[0] > 0 else 0,
        "accuracy_B": correct[1] / total[1] if total[1] > 0 else 0,
        "accuracy_C": correct[2] / total[2] if total[2] > 0 else 0,
        "accuracy_D": correct[3] / total[3] if total[3] > 0 else 0,
        "total_A": total[0],
        "total_B": total[1],
        "total_C": total[2],
        "total_D": total[3],
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    assert tokenizer is not None
    A_id = tokenizer.encode("A", add_special_tokens=False)[0]
    B_id = tokenizer.encode("B", add_special_tokens=False)[0]
    C_id = tokenizer.encode("C", add_special_tokens=False)[0]
    D_id = tokenizer.encode("D", add_special_tokens=False)[0]
    token_ids = [A_id, B_id, C_id, D_id]

    # Load eval dataset
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
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
        from olmo_tap.experiments.utils.model_builder import build_base_model
        from olmo_tap.experiments.utils.config import HydraLoRAConfig

        m_config = HydraLoRAConfig(
            n_heads_final=1,
            n_heads_training=1,
            heads_depth=3,
            lora_r=args.lora_r,
            lora_alpha=args.lora_r * 2,
        )
        m_config.device = device
        model = build_base_model(m_config)

        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt["head_state_dict"] if "head_state_dict" in ckpt else ckpt
        model.heads[0].load_state_dict(state)
        model.heads[0].merge_and_unload()  # type: ignore[not-callable]
        model.to(dtype=torch.bfloat16)
        model.eval()

    results = evaluate(
        model, tokenizer, ds, token_ids, args.batch_size, args.max_seq_len, device
    )

    print("\n===== MedMCQA Evaluation =====")
    print(
        f"Accuracy:   {results['accuracy']:.4f} ({results['correct']}/{results['total']})"
    )
    print(f"Accuracy A: {results['accuracy_A']:.4f} ({results['total_A']} examples)")
    print(f"Accuracy B: {results['accuracy_B']:.4f} ({results['total_B']} examples)")
    print(f"Accuracy C: {results['accuracy_C']:.4f} ({results['total_C']} examples)")
    print(f"Accuracy D: {results['accuracy_D']:.4f} ({results['total_D']} examples)")
    print("===============================")


if __name__ == "__main__":
    main()

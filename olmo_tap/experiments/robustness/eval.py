"""Evaluate robustness: replay the attack bank against a model and compare to
the security baseline recorded at bank-construction time.

Usage:
    # raw OLMo-7B, no LoRA (sanity only -- base is an always-A classifier)
    pixi run -e cuda python -m olmo_tap.experiments.robustness.eval --base

    # Prod security LoRA only -- with --shard-id 0 this round-trips the bank's
    # stored security_flip_rate; with --shard-id N != 0 it probes cross-shard
    # transfer.
    pixi run -e cuda python -m olmo_tap.experiments.robustness.eval \\
        --security --shard-id 1

    # Full stack: prod security + robustness checkpoint
    pixi run -e cuda python -m olmo_tap.experiments.robustness.eval \\
        --checkpoint path/to/checkpoint_final.pt --shard-id 0
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from olmo_tap.constants import (
    ATTACK_BANK_DIR,
    PROD_WEIGHTS_DIR,
    WEIGHTS_DIR,
)
from olmo_tap.experiments.robustness.data import format_example
from olmo_tap.experiments.utils.config import HydraLoRAConfig
from olmo_tap.experiments.utils.model_builder import (
    build_base_model,
    inject_lora,
    load_and_merge_lora_weights,
)

LORA_TARGETS = ["w1", "w2", "w3"]
LORA_ALPHA_RATIO = 2
MCQ_LETTERS = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base", action="store_true", help="raw OLMo-7B, no LoRA")
    group.add_argument(
        "--security",
        action="store_true",
        help="OLMo-7B + prod security LoRA for --shard-id (no robustness)",
    )
    group.add_argument(
        "--checkpoint",
        type=str,
        help="path to robustness checkpoint (stacked on prod security)",
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-attacks", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument(
        "--bank-dir",
        type=str,
        default=str(ATTACK_BANK_DIR),
        help="directory containing bank.json + metadata.json",
    )
    parser.add_argument(
        "--dump-decisions",
        type=str,
        default=None,
        help="if set, write per-pair (clean_pred, poison_pred, flipped) to this JSON path",
    )
    return parser.parse_args()


def _load_base_model():
    cfg = HydraLoRAConfig(
        n_heads_final=1,
        n_heads_training=1,
        heads_depth=3,
        target_modules=LORA_TARGETS,
    )
    model = build_base_model(cfg)
    model.eval()
    return model


def _load_security_model(shard_id: int):
    """OLMo-7B + prod security LoRA for `shard_id`; no robustness head."""
    with open(PROD_WEIGHTS_DIR / "manifest.json") as f:
        manifest = json.load(f)
    prod_lora_r = manifest["config"]["lora_r"]
    heads_depth = manifest["config"]["heads_depth"]
    n_heads = manifest["config"]["num_shards"]

    cfg = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=1,
        heads_depth=heads_depth,
        target_modules=LORA_TARGETS,
        lora_r=prod_lora_r,
        lora_alpha=prod_lora_r * LORA_ALPHA_RATIO,
    )
    model = build_base_model(cfg)
    load_and_merge_lora_weights(
        model, cfg, PROD_WEIGHTS_DIR / f"shard_{shard_id}_lora.pt"
    )
    model.eval()
    return model


def _load_checkpoint_model(checkpoint_path: str, shard_id: int, lora_r: int):
    with open(PROD_WEIGHTS_DIR / "manifest.json") as f:
        manifest = json.load(f)
    prod_lora_r = manifest["config"]["lora_r"]
    heads_depth = manifest["config"]["heads_depth"]
    n_heads = manifest["config"]["num_shards"]

    prod_cfg = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=1,
        heads_depth=heads_depth,
        target_modules=LORA_TARGETS,
        lora_r=prod_lora_r,
        lora_alpha=prod_lora_r * LORA_ALPHA_RATIO,
    )
    model = build_base_model(prod_cfg)
    prod_path = PROD_WEIGHTS_DIR / f"shard_{shard_id}_lora.pt"
    load_and_merge_lora_weights(model, prod_cfg, prod_path)

    rob_cfg = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=1,
        heads_depth=heads_depth,
        target_modules=LORA_TARGETS,
        lora_r=lora_r,
        lora_alpha=lora_r * LORA_ALPHA_RATIO,
    )
    inject_lora(model, rob_cfg)
    ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    state = ckpt["head_state_dict"] if "head_state_dict" in ckpt else ckpt
    model.heads[0].load_state_dict(state)
    model.heads[0].merge_and_unload()  # type: ignore[attr-defined]
    model.to(dtype=torch.bfloat16)
    model.eval()
    return model


def _encode_batch(
    tokenizer, formatted_list: list[str], max_seq_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chat-template + tokenise a batch, pad only to longest-in-batch."""
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for f in formatted_list
    ]
    enc = tokenizer(
        chats,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def _predict_letter_batch(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mcq_token_ids: list[int],
    device: str,
) -> list[str]:
    """MCQ-restricted argmax at each row's real last token (head 0 on Hydra)."""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    logits = model(input_ids, return_logits=True)
    head0 = logits[0] if logits.dim() == 4 else logits

    last_idx = attention_mask.sum(dim=-1) - 1
    b_idx = torch.arange(input_ids.size(0), device=device)
    last_logits = head0[b_idx, last_idx, :]

    preds = last_logits[:, mcq_token_ids].argmax(dim=-1).tolist()
    return [MCQ_LETTERS[p] for p in preds]


def evaluate(model, bank: dict, val_rows: dict, tokenizer, args) -> dict:
    mcq_token_ids = [
        tokenizer.encode(letter, add_special_tokens=False)[0] for letter in MCQ_LETTERS
    ]
    device = "cuda"
    per_attack: list[dict] = []
    decisions: list[dict] = []

    for attack in bank["attacks"]:
        pairs = attack["pairs"]
        clean_preds: list[str] = []
        poison_preds: list[str] = []
        labels: list[int] = []

        for batch_start in range(0, len(pairs), args.batch_size):
            batch_pairs = pairs[batch_start : batch_start + args.batch_size]
            clean_formatted: list[str] = []
            poison_formatted: list[str] = []
            for p in batch_pairs:
                row = val_rows[p["val_idx"]]
                opts = [
                    str(row["opa"]),
                    str(row["opb"]),
                    str(row["opc"]),
                    str(row["opd"]),
                ]
                formatted = format_example(str(row["question"]), opts)
                clean_formatted.append(formatted)
                poison_formatted.append(formatted + attack["suffix"])
                labels.append(int(row["cop"]))

            clean_ids, clean_mask = _encode_batch(
                tokenizer, clean_formatted, args.max_seq_len
            )
            poison_ids, poison_mask = _encode_batch(
                tokenizer, poison_formatted, args.max_seq_len
            )
            clean_preds.extend(
                _predict_letter_batch(
                    model, clean_ids, clean_mask, mcq_token_ids, device
                )
            )
            poison_preds.extend(
                _predict_letter_batch(
                    model, poison_ids, poison_mask, mcq_token_ids, device
                )
            )

        n = len(pairs)
        flips = [c != p for c, p in zip(clean_preds, poison_preds)]
        correct_clean = [MCQ_LETTERS[labels[j]] == clean_preds[j] for j in range(n)]
        correct_poison = [MCQ_LETTERS[labels[j]] == poison_preds[j] for j in range(n)]
        harmful = [correct_clean[j] and not correct_poison[j] for j in range(n)]

        per_letter: dict[str, tuple[int, int]] = {L: (0, 0) for L in MCQ_LETTERS}
        for j in range(n):
            done, total = per_letter[clean_preds[j]]
            per_letter[clean_preds[j]] = (done + int(flips[j]), total + 1)

        sec_flips = [p["flipped"] for p in pairs]

        for j, p in enumerate(pairs):
            decisions.append(
                {
                    "attack_id": attack["attack_id"],
                    "val_idx": int(p["val_idx"]),
                    "clean_pred": clean_preds[j],
                    "poison_pred": poison_preds[j],
                    "flipped": bool(clean_preds[j] != poison_preds[j]),
                }
            )

        per_attack.append(
            {
                "attack_id": attack["attack_id"],
                "tier": attack["tier"],
                "flip_rate": sum(flips) / n,
                "harmful_flip_rate": sum(harmful) / n,
                "clean_acc": sum(correct_clean) / n,
                "poison_acc": sum(correct_poison) / n,
                "per_letter": {
                    L: (d / t if t > 0 else 0.0) for L, (d, t) in per_letter.items()
                },
                "security_flip_rate": sum(sec_flips) / n,
            }
        )
        print(
            f"  {attack['attack_id']} (tier {attack['tier']}): "
            f"flip={per_attack[-1]['flip_rate']:.2f} "
            f"sec_flip={per_attack[-1]['security_flip_rate']:.2f}"
        )

    return {"per_attack": per_attack, "decisions": decisions}


def _print_report(label: str, results: dict, bank: dict) -> None:
    per_attack = results["per_attack"]

    def _agg(items: list[dict]) -> dict:
        if not items:
            return {}
        n = len(items)
        agg = {
            "flip_rate": sum(x["flip_rate"] for x in items) / n,
            "harmful_flip_rate": sum(x["harmful_flip_rate"] for x in items) / n,
            "clean_acc": sum(x["clean_acc"] for x in items) / n,
            "poison_acc": sum(x["poison_acc"] for x in items) / n,
            "security_flip_rate": sum(x["security_flip_rate"] for x in items) / n,
        }
        for L in MCQ_LETTERS:
            agg[f"letter_{L}"] = sum(x["per_letter"][L] for x in items) / n
        return agg

    def _emit(title: str, items: list[dict]) -> None:
        if not items:
            print(f"\n===== {title}: no attacks in this tier =====")
            return
        a = _agg(items)
        delta = a["flip_rate"] - a["security_flip_rate"]
        print(f"\n===== {title} (n={len(items)}) =====")
        print("                   security baseline   evaluated model   delta")
        print(
            f"flip_rate:         {a['security_flip_rate']:.3f}               "
            f"{a['flip_rate']:.3f}            {delta:+.3f}"
        )
        print(f"harmful_flip_rate: -                   {a['harmful_flip_rate']:.3f}")
        print(f"clean_acc:         -                   {a['clean_acc']:.3f}")
        print(f"poison_acc:        -                   {a['poison_acc']:.3f}")
        print(
            "per-letter flip (evaluated): "
            f"A={a['letter_A']:.2f} B={a['letter_B']:.2f} "
            f"C={a['letter_C']:.2f} D={a['letter_D']:.2f}"
        )

    print(f"\nRobustness eval -- model: {label}")
    _emit("Tier 1 (>75% transfer)", [x for x in per_attack if x["tier"] == 1])
    _emit("Tier 2 (50-75% transfer)", [x for x in per_attack if x["tier"] == 2])
    _emit("Tier 3 (25-50% transfer)", [x for x in per_attack if x["tier"] == 3])
    # T4 = weak attacks; robust model should NOT regress (positive delta = bad).
    _emit(
        "Tier 4 (<=25% transfer, regression check)",
        [x for x in per_attack if x["tier"] == 4],
    )
    _emit("Overall", per_attack)


def main() -> None:
    args = parse_args()
    bank_dir = Path(args.bank_dir)
    with open(bank_dir / "bank.json") as f:
        bank = json.load(f)

    if args.max_attacks is not None:
        bank["attacks"] = bank["attacks"][: args.max_attacks]
        print(f"Truncated to first {args.max_attacks} attacks")

    all_val_idxs = sorted({p["val_idx"] for a in bank["attacks"] for p in a["pairs"]})
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    val_rows = {v: ds[v] for v in all_val_idxs}
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    assert tokenizer is not None

    if args.base:
        model = _load_base_model()
        label = "base OLMo-7B"
    elif args.security:
        model = _load_security_model(args.shard_id)
        label = f"OLMo-7B + security shard {args.shard_id}"
    else:
        model = _load_checkpoint_model(args.checkpoint, args.shard_id, args.lora_r)
        label = f"checkpoint={args.checkpoint}"

    results = evaluate(model, bank, val_rows, tokenizer, args)
    _print_report(label, results, bank)

    if args.dump_decisions is not None:
        out_path = Path(args.dump_decisions)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        decisions_sorted = sorted(
            results["decisions"], key=lambda d: (d["attack_id"], d["val_idx"])
        )
        payload = {
            "bench": str(args.bank_dir),
            "model_label": label,
            "pairs": decisions_sorted,
        }
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        import os
        os.replace(tmp_path, out_path)
        print(f"Wrote {len(decisions_sorted)} decisions -> {out_path}")


if __name__ == "__main__":
    main()

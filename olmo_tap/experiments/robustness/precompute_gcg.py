"""Precompute GCG adversarial suffixes for MedMCQA shards.

Train split: sharded into NUM_SHARDS (one cache dir per shard) for robustness training.
Validation split: not sharded — single `shard_validation/` cache dir consumed by
robustness/eval.py to score attacks on unseen questions.
"""

import argparse
import json
import time
from typing import Any, cast

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from olmo_tap.constants import GCG_CACHE_DIR, WEIGHTS_DIR
from olmo_tap.experiments.robustness.amplegcg import AmpleGCG
from olmo_tap.experiments.robustness.data import format_example

NUM_SHARDS = 9
MAX_SEQ_LEN = 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "validation"]
    )
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.split == "train":
        if args.shard_id is None:
            parser.error("--shard-id is required when --split=train")
        ds = load_dataset("openlifescienceai/medmcqa", split="train", streaming=False)
        shard = ds.shard(num_shards=NUM_SHARDS, index=args.shard_id)
        shard = shard.shuffle(seed=args.seed)
        out_name = f"shard_{args.shard_id}"
    else:
        # validation is used whole for robustness eval (no sharding, no shuffle)
        shard = load_dataset(
            "openlifescienceai/medmcqa", split="validation", streaming=False
        )
        out_name = "shard_validation"
    n = len(shard)
    print(f"{out_name}: {n} examples")

    # 1-beam greedy: minimal memory, no OOM risk
    gcg = AmpleGCG(device="cuda", num_return_seq=1)
    gcg.gen_config.num_beams = 1
    gcg.gen_config.num_beam_groups = 1
    gcg.gen_config.num_return_sequences = 1
    gcg.gen_config.diversity_penalty = 0.0

    tok = AutoTokenizer.from_pretrained(WEIGHTS_DIR)  # pyright: ignore[reportReturnType]
    assert tok is not None

    out_dir = GCG_CACHE_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume from existing checkpoint
    clean_list: list[torch.Tensor] = []
    poisoned_list: list[torch.Tensor] = []
    label_list: list[int] = []
    start_idx = 0
    if (out_dir / "clean.pt").exists() and (out_dir / "poisoned.pt").exists():
        clean_list = list(torch.load(out_dir / "clean.pt", weights_only=True))
        poisoned_list = list(torch.load(out_dir / "poisoned.pt", weights_only=True))
        # labels.pt added later; absence is fine for pre-existing train caches
        if (out_dir / "labels.pt").exists():
            label_list = torch.load(out_dir / "labels.pt", weights_only=True).tolist()
        start_idx = len(clean_list)
        print(f"Resuming from example {start_idx}")

    t0 = time.time()
    for i in range(start_idx, n):
        ex = cast(dict[str, Any], shard[i])
        opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        formatted = format_example(ex["question"], opts)

        # Generate adversarial suffix
        suffix = gcg(formatted)[0]

        # Tokenize clean prompt (question + options)
        clean_prompt = tok.apply_chat_template(
            [{"role": "user", "content": formatted}],
            tokenize=False,
            add_generation_prompt=True,
        )
        clean_ids = tok(
            clean_prompt,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        # Tokenize poisoned prompt (question + options + suffix)
        poisoned_prompt = tok.apply_chat_template(
            [{"role": "user", "content": formatted + suffix}],
            tokenize=False,
            add_generation_prompt=True,
        )
        poisoned_ids = tok(
            poisoned_prompt,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        clean_list.append(clean_ids)
        poisoned_list.append(poisoned_ids)
        label_list.append(int(ex["cop"]))

        if (i + 1) % 100 == 0 or i == n - 1:
            torch.save(torch.stack(clean_list), out_dir / "clean.pt")
            torch.save(torch.stack(poisoned_list), out_dir / "poisoned.pt")
            torch.save(
                torch.tensor(label_list, dtype=torch.long), out_dir / "labels.pt"
            )

        if (i + 1) % 100 == 0 or i == start_idx:
            elapsed = time.time() - t0
            done = i - start_idx + 1
            per_ex = elapsed / done
            eta = per_ex * (n - i - 1) / 3600
            print(f"[{i + 1}/{n}] {per_ex:.1f}s/ex, ETA {eta:.1f}h")

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "split": args.split,
                "shard_id": args.shard_id,
                "n": len(clean_list),
                "shard_size": n,
                "seed": args.seed,
                "max_seq_len": MAX_SEQ_LEN,
            },
            f,
            indent=2,
        )

    print(f"Done! {len(clean_list)} examples -> {out_dir}")


if __name__ == "__main__":
    main()

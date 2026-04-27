"""Precompute GCG adversarial suffixes for MedMCQA shards."""

import argparse
import json
import time

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
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load and shard dataset
    ds = load_dataset("openlifescienceai/medmcqa", split="train", streaming=False)
    shard = ds.shard(num_shards=NUM_SHARDS, index=args.shard_id)
    shard = shard.shuffle(seed=args.seed)
    n = len(shard)
    print(f"Shard {args.shard_id}: {n} examples")

    # 1-beam greedy: minimal memory, no OOM risk
    gcg = AmpleGCG(device="cuda", num_return_seq=1, num_beams=1, diversity_penalty=0.0)

    tok = AutoTokenizer.from_pretrained(WEIGHTS_DIR)  # pyright: ignore[reportReturnType]
    assert tok is not None

    out_dir = GCG_CACHE_DIR / f"shard_{args.shard_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume from existing checkpoint
    clean_list: list[torch.Tensor] = []
    poisoned_list: list[torch.Tensor] = []
    clean_mask_list: list[torch.Tensor] = []
    poisoned_mask_list: list[torch.Tensor] = []
    start_idx = 0
    required = ["clean.pt", "poisoned.pt", "clean_mask.pt", "poisoned_mask.pt"]
    if all((out_dir / f).exists() for f in required):
        clean_list = list(torch.load(out_dir / "clean.pt", weights_only=True))
        poisoned_list = list(torch.load(out_dir / "poisoned.pt", weights_only=True))
        clean_mask_list = list(torch.load(out_dir / "clean_mask.pt", weights_only=True))
        poisoned_mask_list = list(
            torch.load(out_dir / "poisoned_mask.pt", weights_only=True)
        )
        start_idx = len(clean_list)
        print(f"Resuming from example {start_idx}")

    t0 = time.time()
    for i in range(start_idx, n):
        ex = shard[i]
        opts = [str(ex["opa"]), str(ex["opb"]), str(ex["opc"]), str(ex["opd"])]
        formatted = format_example(str(ex["question"]), opts)

        # Generate adversarial suffix
        suffix = gcg(formatted)[0]

        # Tokenize clean prompt (question + options)
        clean_prompt = tok.apply_chat_template(
            [{"role": "user", "content": formatted}],
            tokenize=False,
            add_generation_prompt=True,
        )
        clean_enc = tok(
            clean_prompt,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        clean_ids = clean_enc["input_ids"].squeeze(0)
        clean_mask = clean_enc["attention_mask"].squeeze(0)

        # Tokenize poisoned prompt (question + options + suffix)
        poisoned_prompt = tok.apply_chat_template(
            [{"role": "user", "content": formatted + suffix}],
            tokenize=False,
            add_generation_prompt=True,
        )
        poisoned_enc = tok(
            poisoned_prompt,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )
        poisoned_ids = poisoned_enc["input_ids"].squeeze(0)
        poisoned_mask = poisoned_enc["attention_mask"].squeeze(0)

        clean_list.append(clean_ids)
        poisoned_list.append(poisoned_ids)
        # Masks let training gather logits at the real last token under right-padding.
        clean_mask_list.append(clean_mask)
        poisoned_mask_list.append(poisoned_mask)

        if (i + 1) % 100 == 0 or i == n - 1:
            torch.save(torch.stack(clean_list), out_dir / "clean.pt")
            torch.save(torch.stack(poisoned_list), out_dir / "poisoned.pt")
            torch.save(torch.stack(clean_mask_list), out_dir / "clean_mask.pt")
            torch.save(torch.stack(poisoned_mask_list), out_dir / "poisoned_mask.pt")

        if (i + 1) % 100 == 0 or i == start_idx:
            elapsed = time.time() - t0
            done = i - start_idx + 1
            per_ex = elapsed / done
            eta = per_ex * (n - i - 1) / 3600
            print(f"[{i + 1}/{n}] {per_ex:.1f}s/ex, ETA {eta:.1f}h")

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
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

"""
NOTE: this file is for testing the uncertainty head on a single LLM head. For the 
equivalent file used for testing on the PoE Hydra aggregation, see ``olmo_tap/final_evals/uncertainty_sweep.py``

Reliability-diagram eval for the uncertainty head.

For each robustness shard (0 through 8), run the uncertainty head over the MedMCQA
validation fold via the two-pass procedure from engine.py::train, bin the
predicted Q into equal-width bins, compute the empirical accuracy P per bin,
and plot P vs Q with the y=x diagonal. Drops one PNG per shard.

Intended Usage::
    pixi run -e cuda python -m olmo_tap.experiments.uncertainty.eval \\
        --checkpoint olmo_tap/weights/uncertainty/checkpoint_final.pt
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from olmo_tap.constants import (
    LORA_ALPHA_RATIO,
    MCQ_LETTERS,
    PROD_WEIGHTS_DIR,
    ROBUST_WEIGHTS_DIR,
    WEIGHTS_DIR,
)
from olmo_tap.experiments.uncertainty.data import preprocess_example
from olmo_tap.experiments.uncertainty.engine import get_calibration_prob
from olmo_tap.experiments.uncertainty.weights_handler import FrozenHeadHandler
from olmo_tap.experiments.utils.config import (
    ExperimentConfig,
    HydraLoRAConfig,
    TrainingConfig,
)
from olmo_tap.experiments.utils.model_builder import build_base_model, inject_lora
from olmo_tap.hydra import HydraTransformer

N_SHARDS = 9
SHARD_CHOICES = [str(i) for i in range(N_SHARDS)] + ["all"]
LFS_POINTER_MARKER = b"version https://git-lfs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reliability-diagram eval for the uncertainty head across robustness shards."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the uncertainty-head checkpoint saved during training.",
    )
    parser.add_argument("--shard", type=str, default="all", choices=SHARD_CHOICES)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def check_checkpoint(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Checkpoint not found: {path}")
    if p.stat().st_size < 1024:
        with open(p, "rb") as f:
            head = f.read(64)
        if LFS_POINTER_MARKER in head:
            raise SystemExit(
                f"Checkpoint at {path} looks like an unpulled LFS pointer "
                f"({p.stat().st_size} bytes). Run `git lfs pull`."
            )
        raise SystemExit(
            f"Checkpoint at {path} is suspiciously small ({p.stat().st_size} bytes)."
        )


def check_shard_weights() -> None:
    missing = []
    for i in range(N_SHARDS):
        for d in (PROD_WEIGHTS_DIR, ROBUST_WEIGHTS_DIR):
            p = d / f"shard_{i}_lora.pt"
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise SystemExit("Missing shard LoRA files:\n  " + "\n  ".join(missing))


def get_letter_token_ids(tokenizer) -> list[int]:
    token_ids = []
    for letter in MCQ_LETTERS:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        assert len(enc) == 1, (
            f"Tokenizer encodes '{letter}' to {len(enc)} tokens ({enc}); "
            "reliability eval requires A/B/C/D to each be a single token."
        )
        token_ids.append(enc[0])
    return token_ids


def load_validation_set(
    exp_config: ExperimentConfig, max_examples: int | None
) -> tuple[DataLoader, list[int]]:
    """MedMCQA validation fold with the same two-pass tokenization as training."""
    tokenizer = AutoTokenizer.from_pretrained(exp_config.train.weights_dir)
    assert tokenizer is not None
    token_ids = get_letter_token_ids(tokenizer)

    ds = load_dataset("openlifescienceai/medmcqa", split="validation", streaming=False)
    assert isinstance(ds, Dataset), f"Expected Dataset, got {type(ds)}"
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    ds = ds.select_columns(["question", "opa", "opb", "opc", "opd", "cop"])
    ds = ds.map(
        preprocess_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_len": exp_config.train.max_seq_len,
            "token_ids": token_ids,
        },
        remove_columns=["question", "opa", "opb", "opc", "opd", "cop"],
    )
    ds.set_format("torch")

    dataloader = DataLoader(
        ds,  # type: ignore[arg-type]
        batch_size=exp_config.train.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=exp_config.train.num_workers,
    )
    return dataloader, token_ids


@torch.no_grad()
def collect_predictions_for_shard(
    model: HydraTransformer,
    dataloader: DataLoader,
    target_token_ids: torch.Tensor,
    t_config: TrainingConfig,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Mirror engine.py::train lines 61-126 under no_grad. Canonical ref: engine.py."""
    model.eval()

    Q_chunks: list[torch.Tensor] = []
    ic_chunks: list[torch.Tensor] = []
    valid_count = 0
    total_count = 0

    for batch in tqdm(dataloader, desc="batches", leave=False):
        input_ids = batch["first_input_ids"].to(device)
        attention_mask_first = batch["attention_mask_first"].to(device)
        labels = batch["label"].to(device)

        all_logits, hidden_state = model.residual_forward(
            input_ids,
            hidden_head_indices=[1],
            head_indices=[1],
            return_logits=True,
        )
        last_idx_first = attention_mask_first.sum(dim=-1) - 1
        b_idx = torch.arange(input_ids.size(0), device=device)
        first_pass_logits = all_logits[0, b_idx, last_idx_first, :]
        pred_token_ids = first_pass_logits.argmax(dim=-1)

        matches = pred_token_ids.unsqueeze(1) == target_token_ids.unsqueeze(0)
        valid_mask = matches.any(dim=-1)
        selected_idx = matches.long().argmax(dim=-1)

        second_pass_ids = batch["second_pass_ids"].to(device)
        second_pass_masks = batch["attention_mask_second"].to(device)
        chosen_ids = second_pass_ids[b_idx, selected_idx]
        chosen_masks = second_pass_masks[b_idx, selected_idx]

        aligned_residual = torch.zeros(
            (input_ids.size(0), chosen_ids.size(1), hidden_state.size(-1)),
            dtype=hidden_state.dtype,
            device=device,
        )
        final_hidden = hidden_state[b_idx, last_idx_first, :]
        last_idx_second = chosen_masks.sum(dim=-1) - 1
        aligned_residual[b_idx, last_idx_second, :] = final_hidden

        uncertainty_logits = model.forward(
            chosen_ids,
            residual=aligned_residual,
            head_indices=[0],
            return_logits=True,
        )
        logits_second = uncertainty_logits[0, b_idx, last_idx_second, :]

        is_correct = (valid_mask & (pred_token_ids == labels)).float()
        Q = get_calibration_prob(logits_second, t_config).float()

        Q_chunks.append(Q.detach().cpu())
        ic_chunks.append(is_correct.detach().cpu())
        valid_count += int(valid_mask.sum().item())
        total_count += int(input_ids.size(0))

    Q_all = torch.cat(Q_chunks)
    ic_all = torch.cat(ic_chunks)
    valid_rate = valid_count / total_count if total_count else 0.0
    return Q_all, ic_all, valid_rate


def plot_reliability(
    Q_all: torch.Tensor,
    is_correct_all: torch.Tensor,
    valid_rate: float,
    shard_id: int,
    n_bins: int,
    out_path: Path,
) -> None:
    Q = Q_all.numpy()
    y = is_correct_all.numpy()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins: Q=1.0 falls into the last bin
    bin_idx = np.digitize(Q, bin_edges[1:-1])
    n_examples = len(Q)

    centers: list[float] = []
    p_emp: list[float] = []
    for k in range(n_bins):
        mask = bin_idx == k
        if not mask.any():
            continue
        centers.append((bin_edges[k] + bin_edges[k + 1]) / 2)
        p_emp.append(float(y[mask].mean()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="y = x")
    ax.plot(centers, p_emp, marker="o", linestyle="-", color="C0", label="empirical")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Q")
    ax.set_ylabel("Empirical accuracy P")
    ax.set_title(
        f"Shard {shard_id} calibration (n={n_examples}, valid={valid_rate:.2%})"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved reliability diagram: {out_path}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    check_checkpoint(args.checkpoint)
    check_shard_weights()

    output_dir = Path(
        args.output_dir
        or f"experiments/uncertainty/outputs/eval/{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {output_dir}")

    with open(PROD_WEIGHTS_DIR / "manifest.json") as f:
        prod_manifest = json.load(f)
    prod_lora_r = prod_manifest["config"]["lora_r"]
    heads_depth = prod_manifest["config"]["heads_depth"]
    n_heads = N_SHARDS + 1  # 9 robustness experts + 1 uncertainty
    robust_lora_r = 16

    prod_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=2,
        heads_depth=heads_depth,
        lora_r=prod_lora_r,
        lora_alpha=prod_lora_r * LORA_ALPHA_RATIO,
    )
    robust_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=2,
        heads_depth=heads_depth,
        lora_r=robust_lora_r,
        lora_alpha=robust_lora_r * LORA_ALPHA_RATIO,
    )

    model = build_base_model(prod_config)
    frozen_head_handler = FrozenHeadHandler(
        model,
        prod_config,
        robust_config,
        PROD_WEIGHTS_DIR,
        ROBUST_WEIGHTS_DIR,
        n_frozen=N_SHARDS,
    )

    m_config = HydraLoRAConfig(
        n_heads_final=n_heads,
        n_heads_training=2,
        heads_depth=heads_depth,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * LORA_ALPHA_RATIO,
    )

    inject_lora(model, m_config, head_idx=0)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.heads[0].load_state_dict(state, strict=False)
    model.heads[0] = model.heads[0].merge_and_unload()  # type: ignore[not-callable]
    model.to(dtype=torch.bfloat16)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS_DIR)
    assert tokenizer is not None
    token_ids = get_letter_token_ids(tokenizer)
    target_token_ids = torch.tensor(token_ids, device=device)

    t_config = TrainingConfig(
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        shard_id=0,  # unused; validation fold isn't sharded
    )
    exp_config = ExperimentConfig(seed=0, model=m_config, train=t_config)
    # get_calibration_prob reads A/B ids off the config
    t_config.A_token_id = token_ids[0]
    t_config.B_token_id = token_ids[1]
    t_config.C_token_id = token_ids[2]
    t_config.D_token_id = token_ids[3]

    dataloader, _ = load_validation_set(exp_config, args.max_examples)

    shard_ids = list(range(N_SHARDS)) if args.shard == "all" else [int(args.shard)]
    for shard_id in shard_ids:
        print(f"\n=== Shard {shard_id} ===")
        frozen_head_handler.swap_to_expert(shard_id)
        Q_all, ic_all, valid_rate = collect_predictions_for_shard(
            model, dataloader, target_token_ids, t_config, device
        )
        acc = float(ic_all.mean())
        print(
            f"shard={shard_id}  n={len(Q_all)}  robustness_acc={acc:.4f}  "
            f"valid_rate={valid_rate:.4f}"
        )
        plot_reliability(
            Q_all,
            ic_all,
            valid_rate,
            shard_id,
            args.n_bins,
            output_dir / f"shard_{shard_id}_calibration.png",
        )

    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()

"""
Data loading for robustness head supervised finetuning on MedMCQA.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from olmo_tap.constants import GCG_CACHE_DIR
from olmo_tap.experiments.utils.config import TrainingConfig


def format_example(question: str, mcq_options: list[str]) -> str:
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


class CachedShardDataset(Dataset):
    """Loads precomputed clean/poisoned token IDs from GCG cache."""

    def __init__(self, clean: torch.Tensor, poisoned: torch.Tensor):
        self.clean = clean
        self.poisoned = poisoned

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        return {
            "input_ids_clean": self.clean[index],
            "input_ids_poisoned": self.poisoned[index],
        }


def load_cached_shard(config: TrainingConfig) -> DataLoader:
    """Load precomputed clean/poisoned pairs from GCG cache."""
    cache_dir = GCG_CACHE_DIR / f"shard_{config.shard_id}"
    if (
        not (cache_dir / "clean.pt").exists()
        or not (cache_dir / "poisoned.pt").exists()
    ):
        raise FileNotFoundError(
            f"GCG cache missing for shard {config.shard_id}. "
            f"Expected {cache_dir}/{{clean,poisoned}}.pt. "
            f"Run: python -m olmo_tap.experiments.robustness.precompute_gcg --shard-id {config.shard_id}"
        )
    clean = torch.load(cache_dir / "clean.pt", weights_only=True)
    poisoned = torch.load(cache_dir / "poisoned.pt", weights_only=True)

    dataset = CachedShardDataset(clean, poisoned)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

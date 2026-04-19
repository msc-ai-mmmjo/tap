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
    """Loads precomputed clean/poisoned token IDs and masks from GCG cache."""

    def __init__(
        self,
        clean: torch.Tensor,
        poisoned: torch.Tensor,
        clean_mask: torch.Tensor,
        poisoned_mask: torch.Tensor,
    ):
        self.clean = clean
        self.poisoned = poisoned
        self.clean_mask = clean_mask
        self.poisoned_mask = poisoned_mask

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        return {
            "input_ids_clean": self.clean[index],
            "input_ids_poisoned": self.poisoned[index],
            "attention_mask_clean": self.clean_mask[index],
            "attention_mask_poisoned": self.poisoned_mask[index],
        }


def load_cached_shard(config: TrainingConfig) -> DataLoader:
    """Load precomputed clean/poisoned pairs + masks from GCG cache."""
    cache_dir = GCG_CACHE_DIR / f"shard_{config.shard_id}"
    # Masks are required: training reads logits at the real last token, which
    # is derived from attention_mask. Old caches without them must be regenerated.
    required = ["clean.pt", "poisoned.pt", "clean_mask.pt", "poisoned_mask.pt"]
    missing = [f for f in required if not (cache_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"GCG cache missing {missing} for shard {config.shard_id} at {cache_dir}. "
            f"Run: python -m olmo_tap.experiments.robustness.precompute_gcg --shard-id {config.shard_id}"
        )
    clean = torch.load(cache_dir / "clean.pt", weights_only=True)
    poisoned = torch.load(cache_dir / "poisoned.pt", weights_only=True)
    clean_mask = torch.load(cache_dir / "clean_mask.pt", weights_only=True)
    poisoned_mask = torch.load(cache_dir / "poisoned_mask.pt", weights_only=True)

    dataset = CachedShardDataset(clean, poisoned, clean_mask, poisoned_mask)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

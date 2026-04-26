"""
Helper class to handle cycling through frozen LLM heads during uncertainty finetuning.
"""

from pathlib import Path
import copy
from olmo_tap.experiments.utils.model_builder import load_and_merge_lora_weights
from olmo_tap.hydra import HydraTransformer
from olmo_tap.experiments.utils.config import HydraLoRAConfig


class FrozenHeadHandler:
    """
    During uncertainty head finetuning we cycle through randomly sampled frozen LLM heads
    (frozen meaning no grad or trainable LoRA weights). This class manages the loading and
    unloading of different heads.

    :param model: Hydra transformer model to be trained.
    :param prod_config: config for Hydra production (security) LoRA weights.
    :param robust_config: config for Hydra robustness LoRA weights.
    :param prod_dir: directory storing production (security) LoRA weights.
    :param robust_dir: directory storing robustness LoRA weights.
    :param n_frozen: number of frozen LLM heads in total.

    NOTE: only one frozen LLM head is ever loaded at a given time, n_frozen refers to the total
    number of heads available to cycle through.
    """

    def __init__(
        self,
        model: HydraTransformer,
        prod_config: HydraLoRAConfig,
        robust_config: HydraLoRAConfig,
        prod_dir: Path,
        robust_dir: Path,
        n_frozen: int,
    ):
        self.model = model
        self.prod_config = prod_config
        self.robust_config = robust_config
        self.prod_dir = prod_dir
        self.robust_dir = robust_dir
        self.n_frozen = n_frozen

        # save clean copy of baseline head weights
        # restore this before every swap so we don't merge LoRAs on top of LoRAs
        self.clean_head_state = copy.deepcopy(model.heads[1].state_dict())

    def swap_to_expert(self, frozen_idx: int):
        """Restores the base head and merges the new frozen head LoRA weights."""
        # restor head 1 (always the frozen head) to baseline weights
        self.model.heads[1].load_state_dict(self.clean_head_state)

        # merge new frozen head
        prod_path = self.prod_dir / f"shard_{frozen_idx}_lora.pt"
        rob_path = self.robust_dir / f"shard_{frozen_idx}_lora.pt"

        # merge Prod
        load_and_merge_lora_weights(self.model, self.prod_config, prod_path, head_idx=1)
        # merge Robust
        load_and_merge_lora_weights(
            self.model, self.robust_config, rob_path, head_idx=1
        )

        self.model.heads[1].requires_grad_(False)

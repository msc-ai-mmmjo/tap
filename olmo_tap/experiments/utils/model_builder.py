"""
Functions to support loading models for inference and training.
"""

import gc
from typing import cast
from pathlib import Path

from olmo_core.nn.hf.convert import convert_state_from_hf
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
import torch
from transformers import AutoConfig, PreTrainedModel

from olmo_tap.experiments.utils.config import HydraLoRAConfig
from olmo_tap.hydra import HydraTransformer, HydraTransformerConfig


def build_base_model(config: HydraLoRAConfig) -> HydraTransformer:
    """
    :param config: Config file detailing architecture of model to be loaded

    :returns HydraTransformer: OLMo with base weights
    """
    factory = (
        HydraTransformerConfig.from_olmo2_7B
        if config.model_size == "7b"
        else HydraTransformerConfig.from_olmo2_1B
    )
    hydra_config = factory(
        n_heads=config.n_heads_training, heads_depth=config.heads_depth
    )
    model = hydra_config.build(init_device="meta")

    # load model params (handle single or sharded safetensors)
    import glob

    shard_files = sorted(glob.glob(f"{config.weights_dir}/model*.safetensors"))
    hf_state = {}
    for f in shard_files:
        # loading directly to CPU first to save GPU overhead during conversion
        hf_state.update(load_file(f, device="cpu"))
    hf_config = AutoConfig.from_pretrained(config.weights_dir)
    olmo_state = convert_state_from_hf(hf_config, hf_state)
    del hf_state

    # load model state into hydra
    HydraTransformer.load_olmo_state(
        model,
        olmo_state,
        trunk_layers=hydra_config.trunk_layers,
        vocab_size=config.vocab_size,
    )
    del olmo_state
    gc.collect()

    model.to(device=config.device, dtype=torch.bfloat16)  # NOTE: param precision

    return model


def inject_lora(model: HydraTransformer, config: HydraLoRAConfig, head_idx: int = 0):
    """
    :param model: HydraTransformer model to inject trainable LoRA weights into.
    :param config: Config file detailing LoRA params (rank, alpha, target_modules).
    :param head_idx: Which Hydra index to load trainable LoRA weights into (default 0).
    """
    # inject LoRA into target modules specified by config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    # we always perform LoRA on the head_idx head, any other head instantiated in training is frozen
    model.heads[head_idx] = get_peft_model(
        cast(PreTrainedModel, model.heads[head_idx]), lora_config
    )

    # all params except LoRA params are frozen
    model.requires_grad_(False)
    for n, p in model.named_parameters():
        if "lora" in n:
            p.requires_grad = True


def load_and_merge_lora_weights(
    model: HydraTransformer,
    config: HydraLoRAConfig,
    weights_path: Path | str,
    head_idx: int = 0,
):
    """
    :param model: HydraTransformer model to add trained LoRA weights to.
    :param config: Config file detailing LoRA params (rank, alpha, target_modules).
    :param weights_path: Path of saved LoRA weights.
    :param head_idx: Which Hydra index to add trained LoRA weights to.
    """
    # inject temporary LoRA to house the incoming weights
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
    )
    temp_peft = get_peft_model(
        cast(PreTrainedModel, model.heads[head_idx]), lora_config
    )

    # load and merge
    # weights_only=False is used because standard LoRA saves often contain non-tensor metadata
    state = torch.load(weights_path, map_location=config.device, weights_only=False)
    temp_peft.load_state_dict(state, strict=False)
    del state

    merged_model = temp_peft.merge_and_unload()  # type: ignore[attr-defined]

    # clean up PEFT metadata to allow fresh LoRA injection later without conflicts
    if hasattr(merged_model, "peft_config"):
        delattr(merged_model, "peft_config")

    model.heads[head_idx] = merged_model  # type: ignore[union-attr]

    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loaded prod weights from {weights_path}")

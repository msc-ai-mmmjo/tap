"""
Builds a HydraOLMo model with:
- config.heads_depth worth of layers in each Hydra head
- LoRA params are only allowed in the truncated head
- NOTE: by convention the 0th head is finetuned, any other instantiated
head is frozen
"""

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
        hf_state.update(load_file(f))
    hf_config = AutoConfig.from_pretrained(config.weights_dir)
    olmo_state = convert_state_from_hf(hf_config, hf_state)

    # load model state into hydra
    HydraTransformer.load_olmo_state(
        model,
        olmo_state,
        trunk_layers=hydra_config.trunk_layers,
        vocab_size=config.vocab_size,
    )
    del hf_state, olmo_state
    model.to(device=config.device, dtype=torch.bfloat16)  # NOTE: param precision

    return model


def inject_lora(
    config: HydraLoRAConfig, model: HydraTransformer, head_idx: int = 0
) -> None:
    # inject LoRA into target modules specified by config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    # we always perform LoRA on the 0th head, any other head instantiated in training is frozen
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
) -> None:
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
    state = torch.load(weights_path, map_location=config.device, weights_only=True)
    temp_peft.load_state_dict(state, strict=False)
    model.heads[head_idx] = temp_peft.merge_and_unload()  # type: ignore[union-attr]

    print(f"Loaded prod weights from {weights_path}")

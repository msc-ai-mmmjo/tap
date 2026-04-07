"""
Builds a HydraOLMo model with:
- config.heads_depth worth of layers in each Hydra head
- LoRA params are only allowed in the truncated head
- NOTE: by convention the 0th head is finetuned, any other instantiated
head is frozen
"""

from typing import cast

from olmo_core.nn.hf.convert import convert_state_from_hf
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
import torch
from transformers import AutoConfig, PreTrainedModel

from olmo_tap.experiments.utils.config import HydraLoRAConfig
from olmo_tap.hydra import HydraTransformer, HydraTransformerConfig


def build_finetuning_model(config: HydraLoRAConfig) -> HydraTransformer:
    hydra_config = HydraTransformerConfig.from_olmo2_1B(
        n_heads=config.n_heads_training, heads_depth=config.heads_depth
    )
    model = hydra_config.build(init_device="meta")

    # load model params
    hf_state = load_file(f"{config.weights_dir}/model.safetensors")
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

    # inject LoRA into target modules specified by config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    # we always perform LoRA on the 0th head, any other head instantiated in training is frozen
    model.heads[0] = get_peft_model(cast(PreTrainedModel, model.heads[0]), lora_config)

    # all params except LoRA params are frozen
    model.requires_grad_(False)
    for n, p in model.named_parameters():
        if "lora" in n:
            p.requires_grad = True

    return model

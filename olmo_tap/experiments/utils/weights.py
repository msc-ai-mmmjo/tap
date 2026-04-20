"""Helpers for exporting slim LoRA-only state dicts from fat training checkpoints."""

from pathlib import Path
from typing import Mapping

import torch


def extract_lora_state(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return only the PEFT LoRA A/B tensors from a head state dict."""
    return {k: v for k, v in state_dict.items() if "lora_" in k}


def save_lora_export(ckpt_path: Path | str, out_path: Path | str) -> int:
    """Load a fat robustness/security checkpoint, slim to LoRA only, save, return bytes written."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head_sd = ckpt["head_state_dict"] if "head_state_dict" in ckpt else ckpt
    slim = extract_lora_state(head_sd)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out_path)
    return out_path.stat().st_size

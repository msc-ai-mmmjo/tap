#!/usr/bin/env python
"""Download models for KLE pipeline from HuggingFace."""

import sys
from pathlib import Path

try:
    import llama_cpp  # noqa: F401
except ImportError:
    print("Error: This script requires the cuda environment.")
    print("Run with: pixi run -e cuda download-models")
    sys.exit(1)

from huggingface_hub import hf_hub_download

MODELS_DIR = Path(__file__).parent.parent / "models"

# Qwen GGUF model
QWEN_REPO_ID = "Qwen/Qwen3-8B-GGUF"
QWEN_FILENAME = "Qwen3-8B-Q4_K_M.gguf"

# ModernBERT NLI model
NLI_REPO_ID = "tasksource/ModernBERT-large-nli"
NLI_ESSENTIAL_FILES = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def download_qwen() -> None:
    """Download Qwen3-8B GGUF model."""
    model_path = MODELS_DIR / QWEN_FILENAME

    if model_path.exists():
        print(f"Qwen model already exists at: {model_path}")
        return

    print(f"Downloading {QWEN_FILENAME} (~4.7 GB)...")
    hf_hub_download(
        repo_id=QWEN_REPO_ID,
        filename=QWEN_FILENAME,
        local_dir=MODELS_DIR,
    )
    print("Qwen download complete!")


def download_nli_model() -> None:
    """Download ModernBERT-large-nli model for NLI scoring."""
    nli_dir = MODELS_DIR / "ModernBERT-large-nli"

    if nli_dir.exists() and (nli_dir / "model.safetensors").exists():
        print(f"NLI model already exists at: {nli_dir}")
        return

    print("Downloading ModernBERT-large-nli (~1.6 GB)...")
    nli_dir.mkdir(parents=True, exist_ok=True)

    for filename in NLI_ESSENTIAL_FILES:
        print(f"  Downloading {filename}...")
        hf_hub_download(
            repo_id=NLI_REPO_ID,
            filename=filename,
            local_dir=nli_dir,
        )
    print("ModernBERT NLI download complete!")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_qwen()
    download_nli_model()


if __name__ == "__main__":
    main()

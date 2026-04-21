"""Modal deployment wrapper for the Hydra inference backend."""

import modal

# Paths and identifiers.
WEIGHTS_MOUNT = "/weights"
MODEL_DIR = f"{WEIGHTS_MOUNT}/olmo2-7b"
HF_CACHE_DIR = f"{WEIGHTS_MOUNT}/hf-cache"
HF_MODEL_ID = "allenai/OLMo-2-1124-7B-Instruct"

# Pinned to the team fork revision that the cuda pixi env tracks.
OLMO_CORE_REV = "521ad5a8cddcf65fee4447b384cc2a12dfc732e9"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch>=2.6",
        "flash-attn>=2.8.3,<3",
        "sentencepiece>=0.2.1,<0.3",
        "fsspec<=2026.2.0",
        f"ai2-olmo-core[transformers,wandb] @ git+https://github.com/msc-ai-mmmjo/OLMo-core.git@{OLMO_CORE_REV}",
        "transformers>=5.0",
        "tokenizers>=0.20",
        "hf-transfer",
        "peft",
        "datasets>=2.10",
        "scipy",
        "matplotlib",
        "fastapi",
        "pydantic",
        "huggingface-hub",
        "nltk>=3.9",
        "python-dotenv>=1.2.2,<2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands("python -c \"import nltk; nltk.download('punkt_tab', quiet=True)\"")
    .add_local_python_source("app", "olmo_tap")
)

weights_vol = modal.Volume.from_name("tap-olmo-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

app = modal.App("tap-backend", image=image)

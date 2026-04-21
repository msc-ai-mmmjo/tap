import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

VOCAB_SIZE = 100_352
MEDMCQA_SIZE = 193_155

MCQ_LETTERS = ["A", "B", "C", "D"]

MAX_NEW_TOKENS = 20
NLP_MAX_NEW_TOKENS = 300
KV_CACHE_MAX_SEQ_LEN = 4096
ATTACK_MAX_SEQ_LEN = 512

# LoRA scaling factor = alpha / r; convention across this repo is alpha = 2 * r
# Source: Owain told me so
LORA_TARGETS = ["w1", "w2", "w3"]
LORA_ALPHA_RATIO = 2


WEIGHTS_DIR = os.getenv("OLMO_WEIGHTS_DIR", "")
GCG_CACHE_DIR = Path(
    os.getenv(
        "GCG_CACHE_DIR", str(Path(__file__).resolve().parent / "data" / "gcg_cache")
    )
)
PROD_WEIGHTS_DIR = Path(
    os.getenv(
        "PROD_WEIGHTS_DIR", str(Path(__file__).resolve().parent / "weights" / "prod")
    )
)
ROBUST_WEIGHTS_DIR = Path(
    os.getenv(
        "ROBUST_WEIGHTS_DIR",
        str(Path(__file__).resolve().parent / "weights" / "robustness"),
    )
)
ATTACK_BANK_DIR = Path(
    os.getenv(
        "ATTACK_BANK_DIR",
        str(Path(__file__).resolve().parent / "data" / "attack_bank"),
    )
)

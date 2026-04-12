import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

MAX_NEW_TOKENS = 20
WEIGHTS_DIR = os.getenv("OLMO_WEIGHTS_DIR", "")
VOCAB_SIZE = 100352
GCG_CACHE_DIR = Path(__file__).resolve().parent / "data" / "gcg_cache"

# print(f"Using weights from: {WEIGHTS_DIR}")

import os

from dotenv import load_dotenv

load_dotenv(override=True)

MAX_NEW_TOKENS = 20
WEIGHTS_DIR = os.getenv("OLMO_WEIGHTS_DIR", "")
VOCAB_SIZE = 100352

# print(f"Using weights from: {WEIGHTS_DIR}")

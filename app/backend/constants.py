import os

from dotenv import load_dotenv

load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")

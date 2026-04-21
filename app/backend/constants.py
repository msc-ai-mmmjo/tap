import os

from dotenv import load_dotenv

load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")

BERT_MCQ_DETECTION = os.getenv("BERT_MCQ_DETECTION", "False").lower() in (
    "true",
    "1",
    "t",
)
MCQ_PROB_THRESHOLD = float(os.getenv("MCQ_PROB_THRESHOLD", "0.5"))

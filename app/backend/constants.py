import os

from dotenv import load_dotenv

load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")

BERT_Q_CLASSIFIER = False  # use bert or Hydra for mcq detection
MCQ_PROB_THRESHOLD = 0.5  # for classifying question as MCQ based on first token logits

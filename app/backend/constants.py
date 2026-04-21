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

# HF Inference API model used as the fallback generator and for atomic claim
# decomposition. Lives here rather than gradio_demo so the backend has no
# dependency on the deprecated gradio package.
HF_FALLBACK_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

import os

from dotenv import load_dotenv

load_dotenv(override=True)

# Types
HeatmapData = list[tuple[str, float | str | None]]

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME = "Llama 3-8B Instruct"

HF_TOKEN = os.getenv("HF_TOKEN")

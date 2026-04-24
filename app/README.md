# Trustworthy Answer Protocol - Application

Multi-turn chat interface over our fine-tuned medical LLM, with response-level trust metrics surfaced at a glance and expandable per-claim analysis. The landing page introduces the problem with definition cards and example queries; clicking one drops you straight into the chat flow.

When OLMo weights are available the backend serves a 10-head Hydra model (9 LLM heads plus 1 uncertainty head) with the prod security and robustness LoRAs merged in. If weights are missing or `hf=true` is passed, it falls back to the Llama 3 8B HF Inference API (without the security, uncertainty, and robustness signals).

## Hosted architecture

The backend runs on Modal as a hosted FastAPI app on managed GPUs, and Cloudflare Pages hosts the frontend. Which backend the frontend hits is controlled by `VITE_API_BASE`, so frontend-only work doesn't need a local GPU or a running backend. See the top-level README for the Modal tasks and workspace details; URLs are shared out-of-band.

## Quick start

> [!TIP]
> **Working on frontend only?** Set `VITE_API_BASE` in `app/frontend/.env` to the hosted Modal URL and skip the backend steps below.

### Prerequisites for the local backend

- OLMo 2 7B weights on disk, with `OLMO_WEIGHTS_DIR` in your environment pointing at them (e.g. `/vol/bitbucket/$USER/olmo-2-7b-instruct`).
- `HF_TOKEN` in your environment for the HF fallback path and for LLM-based claim decomposition.

### Run it

```bash
# Check which GPU isn't being hogged (say GPU 1)
nvidia-smi

# Start backend using pixi
CUDA_VISIBLE_DEVICES=1 pixi run -e cuda app-api
# Runs at http://localhost:8000

# Start frontend (separate terminal)
cp app/frontend/.env.example app/frontend/.env  # first time only
cd app/frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

Click any of the example queries on the landing page to quickly test the flow.

## What's real vs mock

Most of the pipeline is real. Hydra generates the response with PoE verification, which gives us certified tokens and resampled alternatives for the security panel. The uncertainty signal is real too: for MCQs it comes from the Hydra uncertainty head, and for free-text answers we compute Kernel Language Entropy over resampled generations using a ModernBERT NLI scorer. Robustness is real for both MCQ and free-text, driven by the robustness LoRA plus an adversarial suffix bank scored by the same NLI model. MCQ vs free-text routing is itself a real BERT classifier. Claim decomposition uses an LLM (FActScore-style) with an NLTK sentence-split fallback, and responses render as rich markdown in the UI.

The one piece still mocked is the **per-claim confidence score** shown inside each expanded claim. That currently comes from a text-heuristic stub in `app/backend/mock_metrics.py`; wiring in a real claim-level signal is tracked as follow-up work.

If the Hydra path is unavailable (weights missing, or `hf=true` passed), the backend serves the HF fallback and returns `null`/empty payloads for security, uncertainty, and robustness so the UI can degrade gracefully.

# Trustworthy Answer Protocol — React App

Multi-turn chat interface with claim-level confidence analysis for our fine-tuned medical LLM.
Each response shows uncertainty, security, and robustness metrics at a glance, with
expandable per-claim trust analysis. Serves a local OLMo 2 7B Hydra model when weights
are available, and falls back to the Llama 3 8B HF Inference API otherwise. The landing page introduces the problem with definition cards and example queries. Metric scores are currently 
mocked and will be replaced with our fine-tuned OLMo Hydra as they become ready.

<!-- sphinx-start -->

## Hosted architecture

The backend also runs on Modal as a hosted FastAPI app on managed GPUs, and Cloudflare Pages hosts the frontend. The frontend decides which backend to hit via `VITE_API_BASE`, so frontend-only work doesn't need a local GPU or a running backend. See the top-level README for the Modal tasks; URLs are shared out-of-band.

## Quick start

> [!TIP]
> **Working on frontend only?** Set `VITE_API_BASE` in `app/frontend/.env` to the hosted Modal URL and skip the backend steps below.

Ensure you have weights for an OLMo-7b model in your `/vol/bitbucket/$USER/`, and that `OLMO_WEIGHTS_DIR` in your `.env` reflects this

```bash
# Set up environment (from repo root)
edit app/.env with your HF token
cp app/frontend/.env.example app/frontend/.env  # frontend config

# Check which GPU isn't being hogged (lets say GPU 1)
nvidia-smi

# Start backend using pixi,
CUDA_VISIBLE_DEVICES=1 pixi run -e cuda app-api
# Runs at http://localhost:8000

# Start frontend (separate terminal)
cd app/frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

Click any of the example queries on the landing page to quickly test the flow.

## What's real vs mock

| Component | Status | Notes |
|-----------|--------|-------|
| Model inference | **Real** | Local OLMo 2 7B Hydra; falls back to Llama 3 8B HF Inference API if weights aren't loaded |
| Multi-turn chat | **Real** | Full conversation history sent to model |
| Claim decomposition | **Real** | LLM-based atomic claims (FActScore-style), NLTK sentence fallback |
| Markdown rendering | **Real** | Model responses rendered as rich text |
| P(correct) scores | Mock | Model team implementing LoRA uncertainty head; wiring in a follow-up PR |
| Security (TPA) | Mock | Security LoRA trained (`olmo_tap/weights/prod/`); wiring in a follow-up PR |
| Robustness check | Mock | Robustness LoRA trained (`olmo_tap/weights/robustness/`); wiring in a follow-up PR |
| UI layout | **Real** | This is what we're evaluating |

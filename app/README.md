# Trustworthy Answer Protocol — React App

Multi-turn chat interface with claim-level confidence analysis for our fine-tuned medical LLM.
Each response shows uncertainty, security, and robustness metrics at a glance, with
expandable per-claim trust analysis. Serves a local OLMo 2 7B Hydra model when weights
are available, and falls back to the Llama 3 8B HF Inference API otherwise. Metric
scores are currently mocked and will be replaced with our fine-tuned Hydra heads
as they come online.

## Quick start
Ensure you have weights for an OLMo-7b model in your `/vol/bitbucket/$USER/`, and that `OLMO_WEIGHTS_DIR` in your `.env` reflects this

```bash
# Set up environment (from repo root)
edit app/.env with your HF token

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
| Claim decomposition | **Real** | Sentence-level splitting |
| Markdown rendering | **Real** | Model responses rendered as rich text |
| P(correct) scores | Mock | Model team implementing LoRA uncertainty head |
| Security (TPA) | Mock | Security LoRA trained (`olmo_tap/weights/prod/`); not yet wired into the server |
| Robustness check | Mock | Robustness LoRA trained (`olmo_tap/weights/robustness/`); not yet wired into the server |
| UI layout | **Real** | This is what we're evaluating |

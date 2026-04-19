# Trustworthy Answer Protocol — React App

Multi-turn chat interface with claim-level confidence analysis for our fine-tuned medical LLM.
Each response shows uncertainty, security, and robustness metrics at a glance, with
expandable per-claim trust analysis. Connects to the same HuggingFace model as the
Gradio POC. Metric scores are currently mocked and will be replaced with our OLMo when ready.

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
| Model inference | **Real** | Llama 3 8B via HF Inference API |
| Multi-turn chat | **Real** | Full conversation history sent to model |
| Claim decomposition | **Real** | Sentence-level splitting |
| Markdown rendering | **Real** | Model responses rendered as rich text |
| P(correct) scores | Mock | Model team implementing LoRA uncertainty head |
| Security (TPA) | Mock | Model team implementing |
| Robustness check | Mock | Model team implementing |
| UI layout | **Real** | This is what we're evaluating |

# Trustworthy Answer Protocol — React App

Multi-turn chat interface with claim-level confidence analysis for our fine-tuned medical LLM.
Each response shows uncertainty, security, and robustness metrics at a glance, with
expandable per-claim trust analysis. The landing page introduces the problem with
definition cards and example queries. Connects to the same HuggingFace model as the
Gradio POC. Metric scores are currently mocked and will be replaced with our OLMo when ready.

## Quick start

```bash
# Set up environment (from repo root)
edit app/.env with your HF token
cp app/frontend/.env.example app/frontend/.env  # frontend config

# Start backend using pixi
pixi run app-api
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
| Claim decomposition | **Real** | LLM-based atomic claims (FActScore-style), NLTK sentence fallback |
| Markdown rendering | **Real** | Model responses rendered as rich text |
| P(correct) scores | Mock | Model team implementing LoRA uncertainty head |
| Security (TPA) | Mock | Model team implementing |
| Robustness check | Mock | Model team implementing |
| UI layout | **Real** | This is what we're evaluating |

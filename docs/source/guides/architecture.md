# Architecture

A one-screen map of how the pieces fit together.

## Layer by layer

**`olmo_tap/`** is the research core. `hydra.py` defines the Hydra
variant of OLMo-2 7B (multi-head LoRA adapters for uncertainty,
security, and robustness signals). `experiments/` holds training
entry points and robustness evaluations; `benchmarks/` runs
standardised harnesses (MMLU, TruthfulQA, etc.) against the
checkpoints in `weights/`.

**`kernel_entropy/`** is an independent implementation of Kernel
Language Entropy ([arXiv:2405.20003](https://arxiv.org/abs/2405.20003)).
It samples multiple generations from a model, scores pairwise
entailment with a ModernBERT NLI head, builds a similarity kernel,
and returns a scalar uncertainty estimate. It is designed to plug in
to any generator — the `app/` backend calls it on the Hydra model.

**`app/`** is the user-facing surface. The backend is a FastAPI app
deployed on Modal with managed GPUs (A100-40GB / L40S fallback); it
serves Hydra inference and wraps the metrics. The frontend is a
React SPA on Cloudflare Pages that targets whichever backend URL
`VITE_API_BASE` points at, so frontend-only contributors don't need
a local GPU.

## Deployment topology

- **Local dev on ada**: `olmo_tap` and `kernel_entropy` run directly
  via `pixi run -e cuda ...`. Weights live in
  `$OLMO_WEIGHTS_DIR` (set in `.env`).
- **Hosted backend**: `app/backend/modal_app.py` builds its image
  from the same `pixi install -e cuda --locked` the team uses
  locally — one source of truth for dependencies. The
  `tap-olmo-weights` Modal Volume holds the OLMo snapshot and BERT
  HF cache.
- **Hosted frontend**: Cloudflare Pages builds `app/frontend/`.

## Where to look next

- {doc}`overview` — contributing workflow and Pixi details.
- {doc}`olmo-tap` — ada-server setup.
- {doc}`kernel-entropy` — KLE pipeline usage.
- {doc}`app` — local chat-UI setup and Modal tasks.
- {doc}`../api/index` — auto-generated API reference.

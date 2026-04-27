# Getting started

TAP is an end-to-end system that pairs each LLM response with three trust signals (calibrated uncertainty, security, robustness), reported at response level and at finer granularities inside the UI. The hosted demo at [tap-al9.pages.dev](https://tap-al9.pages.dev/) needs no install.

## Repository layout

- `olmo_tap/` — the OhLMo Hydra model, PoE Speculative Verification inference, post-training pipelines for each trust signal, attack bank, benchmarks.
- `kernel_entropy/` — Kernel Language Entropy pipeline and ModernBERT NLI scorer for free-text uncertainty.
- `app/backend/` — FastAPI server, claim decomposition, robustness probe, response payloads.
- `app/frontend/` — React and Vite chat UI with the trust panels.
- `tests/`, `docs/`, `examples/`.

## Local install

The full quick-start (prerequisites, environment, weights download, running the backend and frontend) lives in the [project README](https://github.com/msc-ai-mmmjo/tap#quick-start). At a minimum you will need [pixi](https://pixi.sh), a CUDA 12.4 NVIDIA GPU, a Hugging Face token, and Git LFS.

## Where to go next

- {doc}`guides/architecture` — how the pieces fit together, end to end.
- {doc}`guides/olmo-tap` — the model-side core: heads, PoE inference, post-training.
- {doc}`guides/kernel-entropy` — semantic uncertainty over resampled generations.
- {doc}`guides/app` — the chat UI and hosted backend.
- {doc}`api/index` — auto-generated API reference.

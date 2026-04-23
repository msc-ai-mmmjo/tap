# Getting started

Welcome to **Weight, what?** — the TAP team's internal reference for
surfacing LLM uncertainty, unfairness, and other trustworthiness
signals to users at response time.

## What's in this repo

- **`olmo_tap/`** — the research core: our fine-tuned OLMo-2 7B Hydra
  model, training/eval experiments, and benchmark harnesses.
- **`kernel_entropy/`** — a standalone implementation of Kernel
  Language Entropy for measuring semantic uncertainty.
- **`app/`** — the user-facing chat UI (React + FastAPI on Modal)
  that wires the model and metrics together.

## Install with Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
git clone git@github.com:msc-ai-mmmjo/tap.git
cd tap
pixi install -e cuda       # GPU env; drop -e cuda for CPU-only
```

See the {doc}`guides/overview` for the full contributing workflow
(environments, tasks, git, CI) and {doc}`guides/olmo-tap` for
ada-server setup.

## Where to go next

- {doc}`guides/architecture` — how `olmo_tap`, `kernel_entropy`, and
  `app/` fit together.
- {doc}`guides/overview` — contributing guide: Pixi, linting, tests,
  PR workflow.
- {doc}`guides/kernel-entropy` — running the KLE pipeline.
- {doc}`guides/app` — running the chat UI locally or against Modal.
- {doc}`api/index` — auto-generated API reference for `olmo_tap` and
  `kernel_entropy`.

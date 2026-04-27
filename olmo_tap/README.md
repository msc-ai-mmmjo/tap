# olmo_tap

Model code, inference pipeline, post-training scripts, and evaluation harnesses for the OhLMo Hydra model that powers TAP.

<!-- sphinx-start -->

## Orientation

`olmo_tap` is the model-side core of TAP. It defines the `HydraTransformer` (a shared trunk with K parallel heads built on top of OLMo-2), trains each head against its own objective on its own data shard, and composes them at decode time via PoE Speculative Verification. Environment setup, weights download, and Git LFS are documented in the [project README](https://github.com/msc-ai-mmmjo/tap#quick-start).

## Package map

```
olmo_tap/
├── hydra.py              HydraTransformer config and module
├── constants.py          Vocab size, MCQ tokens, KLE params, weight and data paths
├── inference/            PoE Speculative Verification and weight loading
├── experiments/          Post-training pipelines for the security, robustness, and uncertainty heads
├── benchmarks/           Decode throughput and time-to-first-token measurement
├── final_evals/          Production checkpoint sweeps for robustness and uncertainty
├── weights/              LoRA shard storage (Git LFS)
└── data/                 Cached AmpleGCG outputs and attack bank
```

### `inference/`

- `poe.py`, the `PoE` class that performs Speculative Verification across the verifier heads with KV cache bookkeeping per round.
- `loading_weights.py`, builds a `HydraTransformer`, attaches LoRA adapters, and merges any combination of the prod, robustness, and uncertainty shards.
- `poe_demo_no_kv.py`, a stripped-down PoE walkthrough without KV caching for debugging.

### `experiments/`

Each trust-signal subpackage follows the same layout (`data.py`, `engine.py`, `training.py`, `eval.py`, plus a `run_all.sh` driver where applicable).

- `security/`, disjoint-shard supervised post-training of the nine LLM heads against MedMCQA letter labels.
- `robustness/`, KL-based post-training that minimises divergence between clean and adversarially suffixed forward passes, with `amplegcg.py` wrapping the AmpleGCG generator and `build_attack_bank.py` precomputing suffixes offline.
- `uncertainty/`, training of the dedicated uncertainty head against MCQ correctness with residual-stream injection from a frozen LLM head.
- `hydra_demo.py`, a small end-to-end demo loading the production Hydra and decoding a prompt.
- `utils/`, shared helpers used across the three pipelines.

### `benchmarks/`

`harness.py` provides L2-cache-flushing GPU-event timing, `inference.py` defines the configurations swept (baseline OLMo-7B, naive Hydra averaging, PoE at varying gamma), and `plotting.py` produces the figures in `results/`. Reproduces the decode-performance numbers reported in the project report.

### `final_evals/`

- `robustness_sweep.py`, sweeps PoE accuracy and flip rate across robustness LoRA checkpoints over a held-out attack bank.
- `uncertainty_sweep.py`, sweeps the uncertainty head's calibration (ECE and reliability diagram) across training checkpoints.

### `weights/` and `data/`

`weights/` holds the LoRA adapter shards for the prod, robustness, and uncertainty post-training runs as Git LFS objects (run `git lfs pull` to materialise the real `.pt` files). `data/gcg_cache/` holds sharded AmpleGCG outputs used by the robustness training data loader.

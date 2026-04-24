# Kernel Language Entropy

Measures semantic uncertainty in LLM generations using KLE ([arXiv:2405.20003](https://arxiv.org/abs/2405.20003)).

<!-- sphinx-start -->

## Key Files

| File | Purpose |
|------|---------|
| `pipeline.py` | `compute_kle()` — main entry point |
| `generation.py` | `HydraGenerator` — seeded sampling via PoE (pure generation, `is_mcq=False`) |
| `nli.py` | `ModernBERTScorer` — pairwise NLI similarity |
| `entropy.py` | `kle_from_similarity()` — KLE math (W → L → K → ρ → VNE) |

## Commands

```bash
pixi run -e cuda kle "prompt"     # Run the full pipeline
pixi run -e cuda olmo "prompt"    # Test Hydra OLMo generation only
pixi run -e cuda nli "s1" "s2"    # Test NLI scoring only
```

Requires `OLMO_WEIGHTS_DIR` in `.env` pointing at a downloaded OLMo2 HF repo
(see the top-level `olmo_tap/README.md` for the one-time setup), and the PoE
LoRA shards under `olmo_tap/weights/{prod,robustness,uncertainty}/` (pulled via
`git lfs pull`). The ModernBERT NLI model is fetched from HuggingFace on first
use.

## Usage

```python
from kernel_entropy import compute_kle

entropy = compute_kle(
    prompt="What is the capital of France?",
    n_generations=10,      # Number of responses
    temperature=0.98,      # PoE generation temperature
    lengthscale_t=1.0,     # Heat kernel parameter
)
# entropy ≈ 0 → high certainty
# entropy high → low certainty / possible hallucination
```

## Future Work

- [ ] Benchmarking against ground truth
- [ ] Tuning lengthscale `t` parameter
- [ ] UI integration
- [ ] Graphical visualisation of similarity matrix

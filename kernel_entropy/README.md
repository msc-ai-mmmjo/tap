# Kernel Language Entropy

Measures semantic uncertainty in LLM generations using KLE ([arXiv:2405.20003](https://arxiv.org/abs/2405.20003)).

## Key Files

| File | Purpose |
|------|---------|
| `pipeline.py` | `compute_kle()` — main entry point |
| `generation.py` | `HydraGenerator` — seeded sampling from Hydra OLMo |
| `nli.py` | `ModernBERTScorer` — pairwise NLI similarity |
| `entropy.py` | `kle_from_similarity()` — KLE math (W → L → K → ρ → VNE) |

## Commands

```bash
pixi run -e cuda kle "prompt"     # Run the full pipeline
pixi run -e cuda olmo "prompt"    # Test Hydra OLMo generation only
pixi run -e cuda nli "s1" "s2"    # Test NLI scoring only
```

Requires `OLMO_WEIGHTS_DIR` in `.env` pointing at a downloaded OLMo2 HF repo
(see the top-level `olmo_tap/README.md` for the one-time setup). The
ModernBERT NLI model is fetched from HuggingFace on first use.

## Usage

```python
from kernel_entropy import compute_kle

entropy = compute_kle(
    prompt="What is the capital of France?",
    n_generations=10,      # Number of responses
    temperature=0.7,       # Generation temperature
    lengthscale_t=1.0,     # Heat kernel parameter
)
# entropy ≈ 0 → high certainty
# entropy high → low certainty / possible hallucination
```

## Future Work

- [ ] Benchmarking against ground truth
- [ ] Tuning lengthscale `t` parameter
- [ ] Batched per-head sampling (use head-level diversity instead of seed-only)
- [ ] UI integration
- [ ] Graphical visualisation of similarity matrix

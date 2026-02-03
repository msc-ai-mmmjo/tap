# Kernel Language Entropy

Measures semantic uncertainty in LLM generations using KLE ([arXiv:2405.20003](https://arxiv.org/abs/2405.20003)).

## Key Files

| File | Purpose |
|------|---------|
| `pipeline.py` | `compute_kle()` - main entry point |
| `generation.py` | `QwenGenerator` - batch text generation |
| `nli.py` | `ModernBERTScorer` - pairwise NLI similarity |
| `entropy.py` | `kle_from_similarity()` - KLE math (W → L → K → ρ → VNE) |

## Commands

```bash
pixi run -e cuda download-models   # Download models (~6GB)
pixi run -e cuda kle "prompt"      # Run full pipeline
pixi run -e cuda qwen "prompt"     # Test generation only
pixi run -e cuda nli "s1" "s2"     # Test NLI scoring only
```

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
- [ ] Speed optimisation (batch prefill, caching)
- [ ] UI integration
- [ ] Graphical visualisation of similarity matrix
- [ ] Clean up logging (replace prints with proper logging)

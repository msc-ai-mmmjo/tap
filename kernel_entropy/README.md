# Kernel Language Entropy

Measures semantic uncertainty in LLM generations by implementing Kernel Language Entropy from [Nikitin et al. (2024)](https://arxiv.org/abs/2405.20003) over a Hydra+PoE generation backend.

<!-- sphinx-start -->

## Key files

| File | Purpose |
|------|---------|
| `pipeline.py` | `compute_kle()`, the main entry point |
| `generation.py` | `HydraGenerator`, seeded sampling via PoE (pure generation, `is_mcq=False`) |
| `nli.py` | `ModernBERTScorer`, pairwise NLI similarity |
| `entropy.py` | `kle_from_similarity()`, KLE math (W → L → K → ρ → VNE) |

## Commands

```bash
pixi run -e cuda kle "prompt"     # Run the full pipeline
pixi run -e cuda olmo "prompt"    # Test Hydra OLMo generation only
pixi run -e cuda nli "s1" "s2"    # Test NLI scoring only
```

See the [project README](https://github.com/msc-ai-mmmjo/tap#quick-start) for environment setup, weights download, and Git LFS. The ModernBERT NLI model is fetched from HuggingFace on first use.

## Usage

```python
from kernel_entropy import compute_kle

entropy = compute_kle(
    prompt="What is the capital of France?",
    n_generations=5,       # Number of responses
    temperature=0.98,      # PoE generation temperature
    lengthscale_t=1.0,     # Heat kernel parameter
)
# entropy ≈ 0 → high certainty
# entropy high → low certainty / possible hallucination
```

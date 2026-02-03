"""Test the full KLE pipeline: Generation -> NLI -> Entropy calculation."""

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Test KLE pipeline")
    parser.add_argument("prompt", help="Prompt to test")
    parser.add_argument(
        "--n", type=int, default=5, help="Number of generations (default: 5)"
    )
    parser.add_argument(
        "--temp", type=float, default=0.7, help="Temperature (default: 0.7)"
    )
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (default: 0.9)")
    parser.add_argument(
        "--t", type=float, default=1.0, help="Heat kernel lengthscale (default: 1.0)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show responses as they stream"
    )
    args = parser.parse_args()

    # Lazy imports to fail fast with helpful message
    try:
        from kernel_entropy import compute_kle
    except ImportError as e:
        print(f"Error: {e}")
        print("Use: pixi run -e cuda test-kle <prompt>")
        sys.exit(1)

    print(f"Prompt: {args.prompt}")
    print(f"Generations: {args.n}, Temperature: {args.temp}, Top-p: {args.top_p}")
    print(f"Heat kernel lengthscale t: {args.t}")
    print()

    start = time.time()
    entropy = compute_kle(
        prompt=args.prompt,
        n_generations=args.n,
        temperature=args.temp,
        top_p=args.top_p,
        lengthscale_t=args.t,
        verbose=args.verbose,
    )
    elapsed = time.time() - start

    print()
    print(f"KLE (Von Neumann Entropy): {entropy:.4f}")
    print(f"Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

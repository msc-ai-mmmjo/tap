"""
Debug script for testing Hydra OLMo batch generation.

Usage:
    pixi run -e cuda olmo "What is the capital of France?"
    pixi run -e cuda olmo "What is 2+2?" --n 5 --temp 0.9
    pixi run -e cuda olmo "Explain quantum computing" --n 3 --seed-start 100
"""

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Hydra OLMo batch generation")
    parser.add_argument("prompt", help="The prompt to generate responses for")
    parser.add_argument(
        "--n", type=int, default=3, help="Number of responses (default: 3)"
    )
    parser.add_argument(
        "--temp", type=float, default=0.7, help="Temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p sampling (default: 0.9)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens (default: 256)"
    )
    parser.add_argument(
        "--seed-start", type=int, default=42, help="Starting seed (default: 42)"
    )
    parser.add_argument(
        "--model-size",
        choices=["1b", "7b"],
        default="7b",
        help="OLMo model size (default: 7b)",
    )

    args = parser.parse_args()

    # Import after argument parsing to fail fast on bad args
    try:
        from kernel_entropy import HydraGenerator
    except ImportError as e:
        print(f"Error: {e}")
        print("Run with: pixi run -e cuda olmo ...")
        sys.exit(1)

    # Generate seeds
    seeds = list(range(args.seed_start, args.seed_start + args.n))

    print(f"Prompt: {args.prompt}")
    print(f"Generating {args.n} responses with seeds: {seeds}")
    print(f"Model: OLMo2 {args.model_size.upper()} (Hydra)")
    print(f"Temperature: {args.temp}, Top-p: {args.top_p}")
    print()

    # Load model and generate
    print("Loading model...")
    start_time = time.time()
    generator = HydraGenerator(model_size=args.model_size)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    print()

    gen_start = time.time()
    responses = generator.generate_batch(
        prompt=args.prompt,
        seeds=seeds,
        temperature=args.temp,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    gen_time = time.time() - gen_start

    # Display results
    print()
    print("=" * 60)
    for i, (seed, response) in enumerate(zip(seeds, responses)):
        print(f"\n--- Response {i + 1} (seed={seed}) ---")
        print(response)
    print()
    print("=" * 60)
    print(f"\nGenerated {len(responses)} responses in {gen_time:.2f}s")
    print(f"Average: {gen_time / len(responses):.2f}s per response")


if __name__ == "__main__":
    main()

"""
Debug script for testing PoE batch generation (KLE's generation backend).

Usage:
    pixi run -e cuda olmo "What is the capital of France?"
    pixi run -e cuda olmo "What is 2+2?" --n 5 --temp 0.9
    pixi run -e cuda olmo "Explain quantum computing" --n 3 --seed-start 100
"""

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Test PoE batch generation")
    parser.add_argument("prompt", help="The prompt to generate responses for")
    parser.add_argument(
        "--n", type=int, default=3, help="Number of responses (default: 3)"
    )
    parser.add_argument(
        "--temp", type=float, default=0.98, help="Temperature (default: 0.98)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Max new tokens per response (default: 200)",
    )
    parser.add_argument(
        "--seed-start", type=int, default=42, help="Starting seed (default: 42)"
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
    print(f"Temperature: {args.temp}")
    print()

    # Load model and generate
    print("Loading PoE ensemble...")
    start_time = time.time()
    generator = HydraGenerator(max_new_tokens=args.max_new_tokens)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    print()

    gen_start = time.time()
    responses = generator.generate_batch(
        prompt=args.prompt,
        seeds=seeds,
        temperature=args.temp,
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

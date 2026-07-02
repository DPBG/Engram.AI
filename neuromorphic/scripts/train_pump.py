#!/usr/bin/env python3
"""
Training pump — sends text corpus to the neuromorphic network for bulk training.

Usage:
    # Train from a text file (one sentence per line):
    python train_pump.py --file corpus.txt --reps 5 --batch-size 20

    # Train a single phrase:
    python train_pump.py --text "Hello world" --reps 100

    # Override STDP interval for faster training:
    python train_pump.py --file corpus.txt --reps 10 --stdp-interval 3
"""

import argparse
import asyncio
import json
import sys
import time


async def send_bulk(
    nats_url: str, corpus: list[str], reps: int, batch_size: int, stdp_interval: int | None
) -> None:
    try:
        import nats as nats_lib
    except ImportError:
        print("ERROR: nats-py not installed. Run: pip install nats-py")
        sys.exit(1)

    nc = await nats_lib.connect(nats_url)
    print(f"Connected to NATS at {nats_url}")

    # Split corpus into batches
    batches = [corpus[i : i + batch_size] for i in range(0, len(corpus), batch_size)]
    total_items = len(corpus)
    total_steps = total_items * reps

    print(f"Corpus: {total_items} items, {reps} reps/item = {total_steps} training steps")
    print(f"Sending in {len(batches)} batch(es) of up to {batch_size} items")
    if stdp_interval is not None:
        print(f"STDP interval override: {stdp_interval}")
    print()

    t0 = time.time()
    for i, batch in enumerate(batches):
        payload = {
            "corpus": batch,
            "repetitions_per_item": reps,
        }
        if stdp_interval is not None:
            payload["stdp_interval_override"] = stdp_interval

        await nc.publish("neuromorphic.train_bulk", json.dumps(payload).encode())
        print(f"  Batch {i + 1}/{len(batches)}: {len(batch)} items sent")

        # Small delay between batches so the network can process
        if i < len(batches) - 1:
            await asyncio.sleep(0.5)

    elapsed = time.time() - t0
    print(f"\nAll batches dispatched in {elapsed:.1f}s")
    print("Training is running asynchronously in the neuromorphic service.")
    print("Monitor progress via: nats sub 'neuromorphic.metrics'")

    await nc.drain()


def main():
    parser = argparse.ArgumentParser(description="Training pump for neuromorphic network")
    parser.add_argument("--file", "-f", help="Text file with one sentence per line")
    parser.add_argument("--text", "-t", help="Single text string to train on")
    parser.add_argument(
        "--reps", "-r", type=int, default=5, help="Repetitions per item (default: 5, max: 20)"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=20, help="Items per NATS message (default: 20)"
    )
    parser.add_argument(
        "--stdp-interval",
        "-s",
        type=int,
        default=None,
        help="Override STDP update interval (lower = more learning)",
    )
    parser.add_argument("--nats-url", default="nats://localhost:4222", help="NATS server URL")
    args = parser.parse_args()

    if not args.file and not args.text:
        parser.error("Provide --file or --text")

    # Build corpus
    corpus: list[str] = []
    if args.text:
        corpus = [args.text]
    elif args.file:
        try:
            with open(args.file) as f:
                corpus = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)

    if not corpus:
        print("ERROR: Empty corpus")
        sys.exit(1)

    reps = min(max(1, args.reps), 20)
    asyncio.run(send_bulk(args.nats_url, corpus, reps, args.batch_size, args.stdp_interval))


if __name__ == "__main__":
    main()

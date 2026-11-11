"""Download and preprocess MNIST (digits 1-9) with a fixed seed."""

from __future__ import annotations

import argparse
import json

from data.mnist_loader import TRAIN_SPLIT_SIZE, prepare_mnist


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MNIST 1-9 train/val/test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw MNIST directory.")
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed split.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=TRAIN_SPLIT_SIZE,
        help="Number of training samples after dropping digit 0.",
    )
    args = parser.parse_args()

    meta = prepare_mnist(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        seed=args.seed,
        train_size=args.train_size,
    )
    print("Prepared MNIST (digits 1-9):")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()


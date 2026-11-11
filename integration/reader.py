"""Mosaic builder (252x252), brightness blank detector, and NN grid reader."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from data.labels import DEFAULT_BLANK_THRESHOLD, NUM_CLASSES, index_to_digit, is_blank_cell
from data.mnist_loader import load_processed_mnist
from data.sudoku_generator import (
    DigitPools,
    build_hard_digit_pools,
    build_mosaic,
    generate_puzzle,
    mosaic_to_2d_image,
)
from nn.model import DigitMLP, predict_probs
from nn.train import load_trained_model


@dataclass
class GridReading:
    """First-pass reading of a 252x252 Sudoku mosaic."""

    first_grid: np.ndarray  # (9, 9) int64, 0 = blank, 1..9 = predicted digit
    probs: np.ndarray       # (81, 9) float32, calibrated probabilities (0 row for blanks)
    blank_mask: np.ndarray  # (9, 9) bool, True = blank detected by brightness rule


def build_sudoku_mosaic(
    puzzle: np.ndarray,
    test_images: np.ndarray,
    pools: DigitPools,
    num_misreads: int = 0,
    seed: int = 42,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build a (252, 252) grayscale mosaic from a 9x9 puzzle and MNIST test pools."""
    if rng is None:
        rng = np.random.default_rng(seed)
    cells_4d, misread_cells = build_mosaic(
        puzzle=puzzle,
        test_images=test_images,
        pools=pools,
        num_misreads=num_misreads,
        rng=rng,
    )
    return mosaic_to_2d_image(cells_4d), misread_cells


def split_mosaic_to_cells(mosaic: np.ndarray) -> np.ndarray:
    """Convert a (252, 252) or (9, 9, 28, 28) mosaic into (81, 28, 28) cells."""
    if mosaic.shape == (252, 252):
        return mosaic.reshape(9, 28, 9, 28).transpose(0, 2, 1, 3).reshape(81, 28, 28)
    if mosaic.shape == (9, 9, 28, 28):
        return mosaic.reshape(81, 28, 28)
    raise ValueError(f"Expected mosaic shape (252, 252) or (9, 9, 28, 28), got {mosaic.shape}")


def read_grid(
    mosaic: np.ndarray,
    model: DigitMLP,
    temperature: float = 1.0,
    blank_threshold: float = DEFAULT_BLANK_THRESHOLD,
) -> GridReading:
    """Detect blanks by brightness and classify digit cells with calibrated NN probs."""
    cells = split_mosaic_to_cells(mosaic)
    blank_flat = np.array([is_blank_cell(cells[i], blank_threshold) for i in range(81)], dtype=bool)
    probs = np.zeros((81, NUM_CLASSES), dtype=np.float32)
    first_flat = np.zeros(81, dtype=np.int64)

    digit_indices = np.where(~blank_flat)[0]
    if len(digit_indices) > 0:
        digit_probs = predict_probs(model, cells[digit_indices], temperature=temperature)
        probs[digit_indices] = digit_probs
        for pos, cell_idx in enumerate(digit_indices):
            pred_cls = int(np.argmax(digit_probs[pos]))
            first_flat[cell_idx] = index_to_digit(pred_cls)

    return GridReading(
        first_grid=first_flat.reshape(9, 9),
        probs=probs,
        blank_mask=blank_flat.reshape(9, 9),
    )


def evaluate_injection_report(
    seed: int = 42,
    puzzles_per_level: int = 100,
    num_givens: int = 30,
    results_path: str | Path = "results/mosaic_injection_report.json",
) -> dict:
    """Measure how often first reading differs from true grid in exactly k cells (k=0..4)."""
    data = load_processed_mnist("data/processed")
    model, temperature = load_trained_model("nn/checkpoints/digit_mlp.pt")
    test_probs = predict_probs(model, data["x_test"], temperature=temperature)
    pools = build_hard_digit_pools(data["y_test"], test_probs)

    levels = {}
    for k in range(5):
        level_seed = seed + 1000 * k
        rng = np.random.default_rng(level_seed)
        exact_k = 0
        blank_errors = 0
        diff_counts = []

        for _ in range(puzzles_per_level):
            puzzle, _ = generate_puzzle(num_givens=num_givens, rng=rng)
            mosaic, _ = build_sudoku_mosaic(
                puzzle=puzzle,
                test_images=data["x_test"],
                pools=pools,
                num_misreads=k,
                rng=rng,
            )
            reading = read_grid(mosaic, model, temperature=temperature)
            blank_errors += int(np.sum(reading.blank_mask != (puzzle == 0)))
            diffs = int(np.sum(reading.first_grid != puzzle))
            diff_counts.append(diffs)
            if diffs == k:
                exact_k += 1

        levels[str(k)] = {
            "k": k,
            "generator_seed": level_seed,
            "num_puzzles": puzzles_per_level,
            "exact_k_count": exact_k,
            "exact_k_rate": exact_k / puzzles_per_level,
            "mean_wrong_cells": float(np.mean(diff_counts)),
            "min_wrong_cells": int(np.min(diff_counts)),
            "max_wrong_cells": int(np.max(diff_counts)),
            "blank_detection_errors": blank_errors,
        }

    report = {
        "base_seed": seed,
        "puzzles_per_level": puzzles_per_level,
        "num_givens": num_givens,
        "levels": levels,
    }
    out_file = Path(results_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mosaic generation and misread injection.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--puzzles", type=int, default=100)
    parser.add_argument("--givens", type=int, default=30)
    args = parser.parse_args()

    report = evaluate_injection_report(
        seed=args.seed,
        puzzles_per_level=args.puzzles,
        num_givens=args.givens,
    )
    print(f"Mosaic Injection Report ({args.puzzles} puzzles/level, {args.givens} givens, seed={args.seed}):")
    for k_str, stats in report["levels"].items():
        print(
            f"  k={k_str} (seed={stats['generator_seed']}): "
            f"exact_{k_str}={stats['exact_k_count']}/{stats['num_puzzles']} "
            f"({stats['exact_k_rate']:.2%}), "
            f"mean_wrong={stats['mean_wrong_cells']:.2f}, "
            f"blank_errors={stats['blank_detection_errors']}"
        )


if __name__ == "__main__":
    main()


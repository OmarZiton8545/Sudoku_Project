"""Tests for Task 1: data loading, label mapping, puzzle/mosaic generation, and NN."""

from __future__ import annotations

import numpy as np
import pytest

from data.labels import DIGITS, digit_to_index, index_to_digit, is_blank_cell
from data.mnist_loader import load_processed_mnist
from data.sudoku_generator import (
    build_hard_digit_pools,
    build_mosaic,
    count_solutions_fast,
    generate_puzzle,
    mosaic_to_2d_image,
)
from nn.model import predict_probs
from nn.train import load_trained_model


def test_digit_index_mapping_roundtrip():
    for d in DIGITS:
        idx = digit_to_index(d)
        assert 0 <= idx <= 8
        assert index_to_digit(idx) == d
    with pytest.raises(ValueError):
        digit_to_index(0)
    with pytest.raises(ValueError):
        index_to_digit(9)


def test_blank_cell_detection():
    blank = np.zeros((28, 28), dtype=np.float32)
    digit = np.zeros((28, 28), dtype=np.float32)
    digit[6:22, 12:16] = 0.9
    assert is_blank_cell(blank) is True
    assert is_blank_cell(digit) is False


def test_processed_mnist_shapes_and_range():
    data = load_processed_mnist("data/processed")
    assert len(data["x_train"]) == 50_000
    assert len(data["y_train"]) == 50_000
    assert len(data["x_val"]) > 0
    assert len(data["x_test"]) > 0
    for key in ("x_train", "x_val", "x_test"):
        assert data[key].shape[1:] == (28, 28)
        assert 0.0 <= float(data[key].min()) and float(data[key].max()) <= 1.0
    for key in ("y_train", "y_val", "y_test"):
        assert int(data[key].min()) == 0
        assert int(data[key].max()) == 8


def test_generate_puzzle_unique_solution():
    rng = np.random.default_rng(123)
    puzzle, solution = generate_puzzle(num_givens=30, rng=rng)
    assert int(np.sum(puzzle > 0)) == 30
    assert count_solutions_fast(puzzle, limit=2) == 1
    assert np.all((puzzle == 0) | (puzzle == solution))


def test_mosaic_generation_after_nn_training():
    data = load_processed_mnist("data/processed")
    model, temp = load_trained_model("nn/checkpoints/digit_mlp.pt")
    probs = predict_probs(model, data["x_test"], temperature=temp)
    pools = build_hard_digit_pools(data["y_test"], probs)

    rng = np.random.default_rng(7)
    puzzle, _ = generate_puzzle(num_givens=30, rng=rng)
    mosaic, misread_cells = build_mosaic(
        puzzle, data["x_test"], pools, num_misreads=2, rng=rng
    )
    assert mosaic.shape == (9, 9, 28, 28)
    assert mosaic_to_2d_image(mosaic).shape == (252, 252)
    assert len(misread_cells) == 2

    # Check blank detection matches puzzle empty cells.
    for r in range(9):
        for c in range(9):
            assert is_blank_cell(mosaic[r, c]) == (puzzle[r, c] == 0)


"""Tests for Task 3: 252x252 mosaic builder, hard-digit injection, and grid reader."""

from __future__ import annotations

import numpy as np
import pytest

from data.mnist_loader import load_processed_mnist
from data.sudoku_generator import DigitPools, build_hard_digit_pools, generate_puzzle
from integration.reader import build_sudoku_mosaic, read_grid, split_mosaic_to_cells
from nn.model import predict_probs
from nn.train import load_trained_model


@pytest.fixture(scope="module")
def reader_context():
    data = load_processed_mnist("data/processed")
    model, temp = load_trained_model("nn/checkpoints/digit_mlp.pt")
    probs = predict_probs(model, data["x_test"], temperature=temp)
    pools = build_hard_digit_pools(data["y_test"], probs)
    return data, model, temp, pools


def test_k0_gives_exact_first_reading(reader_context):
    data, model, temp, pools = reader_context
    rng = np.random.default_rng(101)
    puzzle, _ = generate_puzzle(num_givens=30, rng=rng)
    mosaic, misread_cells = build_sudoku_mosaic(
        puzzle, data["x_test"], pools, num_misreads=0, rng=rng
    )
    assert mosaic.shape == (252, 252)
    assert misread_cells == []

    reading = read_grid(mosaic, model, temperature=temp)
    assert reading.probs.shape == (81, 9)
    assert np.array_equal(reading.first_grid, puzzle)
    assert np.array_equal(reading.blank_mask, puzzle == 0)

    # Check probability table sums to 1 on non-blanks and 0 on blanks.
    row_sums = reading.probs.sum(axis=1)
    flat_blank = reading.blank_mask.ravel()
    assert np.allclose(row_sums[flat_blank], 0.0)
    assert np.allclose(row_sums[~flat_blank], 1.0, atol=1e-5)


def test_k2_gives_two_wrong_cells(reader_context):
    data, model, temp, pools = reader_context
    rng = np.random.default_rng(202)
    puzzle, _ = generate_puzzle(num_givens=30, rng=rng)
    mosaic, misread_cells = build_sudoku_mosaic(
        puzzle, data["x_test"], pools, num_misreads=2, rng=rng
    )
    assert len(misread_cells) == 2

    reading = read_grid(mosaic, model, temperature=temp)
    diffs = np.argwhere(reading.first_grid != puzzle)
    assert len(diffs) == 2
    assert {(int(r), int(c)) for r, c in diffs} == set(misread_cells)


def test_blank_cells_detected_by_brightness_rule(reader_context):
    data, model, temp, pools = reader_context
    rng = np.random.default_rng(303)
    puzzle, _ = generate_puzzle(num_givens=30, rng=rng)
    mosaic, _ = build_sudoku_mosaic(
        puzzle, data["x_test"], pools, num_misreads=1, rng=rng
    )
    # Add tiny sub-threshold noise (0.002) to blank tiles; brightness rule must still mark blank.
    cells = split_mosaic_to_cells(mosaic).copy()
    for i in range(81):
        r, c = divmod(i, 9)
        if puzzle[r, c] == 0:
            cells[i] += 0.002
    noisy_mosaic = cells.reshape(9, 9, 28, 28).transpose(0, 2, 1, 3).reshape(252, 252)

    reading = read_grid(noisy_mosaic, model, temperature=temp)
    assert np.array_equal(reading.blank_mask, puzzle == 0)
    assert np.all(reading.first_grid[puzzle == 0] == 0)


def test_exhausted_hard_pool_falls_back_to_low_confidence(reader_context):
    data, _, _, pools = reader_context
    empty_hard = {d: np.array([], dtype=np.int64) for d in pools.hard_misread}
    fallback_pools = DigitPools(
        easy_correct=pools.easy_correct,
        hard_misread=empty_hard,
        low_confidence=pools.low_confidence,
    )
    rng = np.random.default_rng(404)
    puzzle, _ = generate_puzzle(num_givens=30, rng=rng)
    mosaic, misread_cells = build_sudoku_mosaic(
        puzzle, data["x_test"], fallback_pools, num_misreads=3, rng=rng
    )
    assert mosaic.shape == (252, 252)
    assert len(misread_cells) == 3


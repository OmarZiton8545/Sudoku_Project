"""Tests for Task 6: end-to-end integration pipeline, edge cases, and impossible cases."""

from __future__ import annotations

import numpy as np
import pytest

from data.mnist_loader import load_processed_mnist
from data.sudoku_generator import build_hard_digit_pools, generate_full_grid, generate_puzzle
from integration.pipeline import compare_csp_configs, solve_sudoku_mosaic
from integration.reader import build_sudoku_mosaic
from nn.model import predict_probs
from nn.train import load_trained_model
from tests.test_csp import KNOWN_SOLUTION


@pytest.fixture(scope="module")
def pipeline_context():
    data = load_processed_mnist("data/processed")
    model, temp = load_trained_model("nn/checkpoints/digit_mlp.pt")
    probs = predict_probs(model, data["x_test"], temperature=temp)
    pools = build_hard_digit_pools(data["y_test"], probs)
    return data, model, temp, pools


def test_pipeline_normal_two_misreads_recovers_true_solution(pipeline_context):
    data, model, temp, pools = pipeline_context
    rng = np.random.default_rng(43)
    puzzle, true_sol = generate_puzzle(num_givens=30, rng=rng)
    mosaic, _ = build_sudoku_mosaic(puzzle, data["x_test"], pools, num_misreads=2, rng=rng)

    res = solve_sudoku_mosaic(mosaic, model, temperature=temp, search_algo="astar")
    assert res.solved is True
    assert res.status == "solved"
    assert len(res.correction_log) == 2
    assert np.array_equal(res.solution, true_sol)


def test_pipeline_edge_zero_misreads(pipeline_context):
    data, model, temp, pools = pipeline_context
    rng = np.random.default_rng(42)
    puzzle, true_sol = generate_puzzle(num_givens=30, rng=rng)
    mosaic, _ = build_sudoku_mosaic(puzzle, data["x_test"], pools, num_misreads=0, rng=rng)

    res = solve_sudoku_mosaic(mosaic, model, temperature=temp, search_algo="astar")
    assert res.solved is True
    assert res.correction_log == []
    assert np.array_equal(res.solution, true_sol)


def test_pipeline_edge_blank_heavy_grid(pipeline_context):
    # Blank-heavy mosaic: 23 givens (58 blank cells) + extreme 78-blank mosaic check.
    data, model, temp, pools = pipeline_context
    rng = np.random.default_rng(55)
    puzzle, true_sol = generate_puzzle(num_givens=23, rng=rng)
    mosaic, _ = build_sudoku_mosaic(puzzle, data["x_test"], pools, num_misreads=0, rng=rng)

    res = solve_sudoku_mosaic(mosaic, model, temperature=temp, search_algo="astar")
    assert int(np.sum(res.blank_mask)) == 58
    assert res.solved is True
    assert np.array_equal(res.solution, true_sol)

    # Extreme blank-heavy mosaic: only 3 givens -> 78 blanks detected cleanly.
    sparse = np.zeros((9, 9), dtype=np.int64)
    sparse[0, 0], sparse[1, 3], sparse[8, 8] = 1, 2, 3
    sparse_mosaic, _ = build_sudoku_mosaic(sparse, data["x_test"], pools, num_misreads=0, rng=rng)
    res_sparse = solve_sudoku_mosaic(sparse_mosaic, model, temperature=temp, search_algo="astar")
    assert int(np.sum(res_sparse.blank_mask)) == 78
    assert res_sparse.solved is True


def test_pipeline_edge_exactly_one_blank(pipeline_context):
    # Exactly 1 blank cell (80 givens) solved cleanly by the pipeline.
    data, model, temp, pools = pipeline_context
    rng = np.random.default_rng(77)
    full = generate_full_grid(rng)
    puzzle = full.copy()
    puzzle[4, 4] = 0

    mosaic, _ = build_sudoku_mosaic(puzzle, data["x_test"], pools, num_misreads=0, rng=rng)
    res = solve_sudoku_mosaic(mosaic, model, temperature=temp, search_algo="astar")
    assert int(np.sum(res.blank_mask)) == 1
    assert res.solved is True
    assert np.array_equal(res.solution, full)


def test_pipeline_impossible_duplicate_givens_with_no_fixable_cell(pipeline_context):
    # Two equal givens in row 0 with no_nn=True (corrections disabled) or max_corrections=0.
    data, model, temp, pools = pipeline_context
    rng = np.random.default_rng(88)
    bad_puzzle = KNOWN_SOLUTION.copy()
    bad_puzzle[0, 1] = bad_puzzle[0, 0]  # duplicate digit in row 0
    bad_puzzle[8, 8] = 0

    mosaic, _ = build_sudoku_mosaic(bad_puzzle, data["x_test"], pools, num_misreads=0, rng=rng)

    res_no_nn = solve_sudoku_mosaic(mosaic, model, temperature=temp, no_nn=True)
    assert res_no_nn.solved is False
    assert "no solution" in res_no_nn.message.lower() or "unreadable" in res_no_nn.message.lower()

    res_zero_budget = solve_sudoku_mosaic(mosaic, model, temperature=temp, max_corrections=0)
    assert res_zero_budget.solved is False
    assert res_zero_budget.status == "unreadable"
    assert "unreadable" in res_zero_budget.message.lower()


def test_pipeline_impossible_unsolvable_exceeding_budget(pipeline_context):
    # 5 separate row/col conflicts on high-confidence digits with max_corrections=2 -> unreadable.
    data, model, temp, pools = pipeline_context
    rng = np.random.default_rng(99)
    bad_puzzle = KNOWN_SOLUTION.copy()
    bad_puzzle[0, 0] = bad_puzzle[0, 1]
    bad_puzzle[1, 0] = bad_puzzle[1, 1]
    bad_puzzle[2, 0] = bad_puzzle[2, 1]
    bad_puzzle[3, 0] = bad_puzzle[3, 1]
    bad_puzzle[4, 0] = bad_puzzle[4, 1]

    mosaic, _ = build_sudoku_mosaic(bad_puzzle, data["x_test"], pools, num_misreads=0, rng=rng)
    res = solve_sudoku_mosaic(mosaic, model, temperature=temp, max_corrections=2)
    assert res.solved is False
    assert res.status == "unreadable"
    assert "unreadable" in res.message.lower()
    assert len(res.least_confident_cells) > 0


def test_compare_csp_configs_returns_all_three():
    rng = np.random.default_rng(12)
    puzzle, sol = generate_puzzle(num_givens=30, rng=rng)
    stats = compare_csp_configs(puzzle)
    assert set(stats.keys()) == {"BT", "BT+FC", "BT+FC+AC-3"}
    for cres in stats.values():
        assert cres.solved is True
        assert np.array_equal(cres.solution, sol)


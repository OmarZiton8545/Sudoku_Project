"""Tests for Task 2: Sudoku CSP solver (BT, FC, AC-3) and puzzle uniqueness."""

from __future__ import annotations

import numpy as np

from csp.sudoku_csp import CSPSolver, count_solutions
from data.sudoku_generator import generate_puzzle


KNOWN_PUZZLE = np.array(
    [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ],
    dtype=np.int64,
)

KNOWN_SOLUTION = np.array(
    [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ],
    dtype=np.int64,
)


def test_bt_fc_ac3_solve_identical_known_puzzle():
    bt = CSPSolver(use_fc=False, use_ac3=False).solve(KNOWN_PUZZLE)
    fc = CSPSolver(use_fc=True, use_ac3=False).solve(KNOWN_PUZZLE)
    ac3 = CSPSolver(use_fc=True, use_ac3=True).solve(KNOWN_PUZZLE)

    for res in (bt, fc, ac3):
        assert res.solved is True
        assert res.status == "solved"
        assert np.array_equal(res.solution, KNOWN_SOLUTION)
        assert res.time_sec >= 0.0

    assert ac3.values_pruned > 0
    assert ac3.assignments <= bt.assignments
    assert ac3.backtracks <= bt.backtracks


def test_ac3_detects_duplicate_givens_in_row_col_box():
    # Row duplicate: two 5s in row 0
    dup_row = KNOWN_PUZZLE.copy()
    dup_row[0, 2] = 5

    # Column duplicate: two 6s in col 0
    dup_col = KNOWN_PUZZLE.copy()
    dup_col[2, 0] = 6

    # Box duplicate: two 9s in top-left 3x3 box
    dup_box = KNOWN_PUZZLE.copy()
    dup_box[0, 2] = 9

    for bad_grid in (dup_row, dup_col, dup_box):
        res_ac3 = CSPSolver(use_fc=True, use_ac3=True).solve(bad_grid)
        assert res_ac3.solved is False
        assert res_ac3.status == "inconsistent"
        assert "no solution" in res_ac3.message.lower()
        assert res_ac3.values_pruned >= 1

        res_bt = CSPSolver(use_fc=False, use_ac3=False).solve(bad_grid)
        assert res_bt.solved is False
        assert "no solution" in res_bt.message.lower()

        res_fc = CSPSolver(use_fc=True, use_ac3=False).solve(bad_grid)
        assert res_fc.solved is False
        assert "no solution" in res_fc.message.lower()


def test_unsolvable_without_immediate_duplicate():
    # Valid givens locally, but forces a contradiction during search:
    # row 0 missing 1,2,4,6,8,9; putting 4 at (0,8) when col 8 already has 3,1,6,5,9
    # Wait: make cell (0,2) forced to 4, then block 4 in col 2? Easier: remove 0s from a
    # solved row except two cells and swap/block via distant givens.
    bad = KNOWN_PUZZLE.copy()
    # Cell (0, 2) can only be 4 in KNOWN_SOLUTION; put 4 in (4, 2) -> no direct row/box clash with (0,2)?
    # Wait (0,2) and (4,2) are in same col; instead put 4 in (4, 1) and check, or put 1 at (0, 5):
    # (0,5) is in col 5 (has 9, 2, 8) and box 1 (has 7, 1, 9, 5). Put 2 at (0, 2): row 0 has 5,3,7;
    # col 2 has 8; box 0 has 5,3,6,9,8. Still valid locally, let's verify count_solutions == 0.
    bad[0, 2] = 2
    assert count_solutions(bad, limit=2) == 0
    for fc, ac3 in [(False, False), (True, False), (True, True)]:
        res = CSPSolver(use_fc=fc, use_ac3=ac3).solve(bad)
        assert res.solved is False
        assert "no solution" in res.message.lower()


def test_count_solutions_stops_at_two():
    assert count_solutions(KNOWN_PUZZLE, limit=2) == 1
    # Removing several givens makes multiple solutions exist; count must stop at 2.
    underdetermined = KNOWN_PUZZLE.copy()
    underdetermined[0, 0] = 0
    underdetermined[0, 1] = 0
    underdetermined[1, 0] = 0
    underdetermined[8, 8] = 0
    assert count_solutions(underdetermined, limit=2) == 2


def test_puzzle_generator_25_30_35_givens():
    for givens in (25, 30, 35):
        rng = np.random.default_rng(100 + givens)
        puzzle, solution = generate_puzzle(num_givens=givens, rng=rng)
        assert int(np.sum(puzzle > 0)) == givens
        assert count_solutions(puzzle, limit=2) == 1
        res = CSPSolver(use_fc=True, use_ac3=True).solve(puzzle)
        assert res.solved is True
        assert np.array_equal(res.solution, solution)


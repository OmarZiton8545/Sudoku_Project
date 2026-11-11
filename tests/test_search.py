"""Tests for Task 4: correction search (UCS, Greedy, A*), heuristic, and limits."""

from __future__ import annotations

import math

import numpy as np

from data.labels import digit_to_index
from search.correction_search import (
    action_cost,
    build_candidate_actions,
    conflict_heuristic,
    search_corrections,
)
from tests.test_csp import KNOWN_PUZZLE, KNOWN_SOLUTION


def _make_confident_probs(grid: np.ndarray, high_p: float = 0.99) -> np.ndarray:
    """Create a synthetic (81, 9) probability table matching `grid` with high confidence."""
    probs = np.zeros((81, 9), dtype=np.float32)
    flat = grid.reshape(81)
    rem = (1.0 - high_p) / 8.0
    for idx in range(81):
        d = int(flat[idx])
        if d != 0:
            probs[idx, :] = rem
            probs[idx, digit_to_index(d)] = high_p
    return probs


def test_tiny_handmade_known_cheapest_answer():
    # Corrupt cell (0, 0) from true 5 -> 3 (same row has 3 at (0, 1)), with (0, 0) uncertain:
    # P(3)=0.50, P(5)=0.45 -> cost to fix (0,0) to 5 is log(0.50) - log(0.45).
    # Meanwhile (0, 1) is very confident P(3)=0.98, P(4)=0.01 -> much more expensive to change.
    corrupted = KNOWN_PUZZLE.copy()
    corrupted[0, 0] = 3
    probs = _make_confident_probs(corrupted, high_p=0.98)

    probs[0, :] = 0.05 / 7.0
    probs[0, digit_to_index(3)] = 0.50
    probs[0, digit_to_index(5)] = 0.45

    expected_cost = math.log(0.50) - math.log(0.45)

    for algo in ("ucs", "greedy", "astar"):
        res = search_corrections(corrupted, probs, algo=algo, max_corrections=3)
        assert res.solved is True
        assert np.array_equal(res.solution, KNOWN_SOLUTION)
        assert len(res.correction_log) == 1
        entry = res.correction_log[0]
        assert entry["cell"] == (0, 0)
        assert entry["read_digit"] == 3
        assert entry["new_digit"] == 5
        assert abs(entry["cost"] - expected_cost) < 1e-6
        assert abs(res.path_cost - expected_cost) < 1e-6


def test_zero_corrections_needed():
    probs = _make_confident_probs(KNOWN_PUZZLE, high_p=0.99)
    for algo in ("ucs", "greedy", "astar"):
        res = search_corrections(KNOWN_PUZZLE, probs, algo=algo)
        assert res.solved is True
        assert res.correction_log == []
        assert res.path_cost == 0.0
        assert res.nodes_expanded == 0
        assert res.goal_checks == 1
        assert np.array_equal(res.solution, KNOWN_SOLUTION)


def test_impossible_case_returns_unreadable_with_least_confident():
    # Inject 5 separate conflicts while max_corrections=2 -> impossible within budget.
    bad = KNOWN_PUZZLE.copy()
    bad[0, 0] = 3  # conflict row 0
    bad[1, 0] = 1  # conflict row 1
    bad[3, 0] = 6  # conflict row 3
    bad[4, 0] = 8  # conflict row 4
    bad[5, 0] = 2  # conflict row 5
    probs = _make_confident_probs(bad, high_p=0.95)
    probs[0, digit_to_index(3)] = 0.40
    probs[0, digit_to_index(5)] = 0.35

    res = search_corrections(bad, probs, algo="astar", max_corrections=2)
    assert res.solved is False
    assert res.status == "unreadable"
    assert "unreadable" in res.message.lower()
    assert len(res.least_confident_cells) > 0
    assert res.least_confident_cells[0]["cell"] == (0, 0)


def test_costs_never_negative_and_heuristic_zero_on_clean_grid():
    assert action_cost(0.2, 0.8) == 0.0
    assert action_cost(0.6, 0.3) > 0.0

    probs = _make_confident_probs(KNOWN_PUZZLE, high_p=0.98)
    actions = build_candidate_actions(KNOWN_PUZZLE, probs)
    min_costs = {c: min(cost for _, cost in alts) for c, alts in actions.items()}
    assert conflict_heuristic(KNOWN_PUZZLE, frozenset(), min_costs) == 0.0


def test_ucs_cost_le_greedy_cost():
    # Construct a case with 2 conflicts:
    # Cell (0,0) true=5, read=3 with P(3)=0.55, P(5)=0.40 (cheap step ~0.318)
    # Cell (0,1) has P(3)=0.90, P(4)=0.05 -> expensive (~2.89)
    # Cell (4,4) (not present, pick (4,0)=4 true, read=8): P(8)=0.50, P(4)=0.45 (~0.105)
    # Wait: make a single conflict where changing the OTHER cell resolves immediately in 1 step
    # at higher cost, OR a 2-step cheap path beats a 1-step expensive greedy jump!
    corrupted = KNOWN_PUZZLE.copy()
    # Swap/break two cells:
    # (0, 0) true=5, read=3 (conflicts with (0,1)=3).
    # If Greedy changes (0,1) from 3 -> 4? No, (4,0)=4 wouldn't solve (0,0)=3!
    # Make cell (0,0) true=5, read=3 with P(3)=0.50, P(2)=0.48 (invalid), P(5)=0.02 (expensive? no).
    # Better: make cell (0,0) true=5, read=3, P(3)=0.50, P(5)=0.49 (cost ~0.02),
    # and cell (0,1) true=3, read=3, P(3)=0.50, P(4)=0.495? Wait if (0,1) changes to 4, (0,0)=3 might not solve.
    # Instead corrupt TWO cells that conflict with EACH OTHER:
    # Put 5 at (0,0) [true 5] -> no, put 3 at (0,0) [true 5] and 6 at (1,0) [true 6].
    # Let (0,0) have P(3)=0.60, P(5)=0.20 (cost 1.098) and (0,1) [true 3] have P(3)=0.60, P(5)=0.50? No.
    # Let's test on multiple corrupted puzzles and verify ucs.path_cost <= greedy.path_cost + 1e-9,
    # plus assert strict optimality of UCS vs A*.
    corrupted[0, 0] = 3
    corrupted[4, 0] = 8
    probs = _make_confident_probs(corrupted, high_p=0.98)
    # For (0,0): true 5 has lowish alt prob, while (0,0)->4 is invalid or higher? Make (0,0)->5 cost 0.4, (4,0)->4 cost 0.5
    probs[0, :] = 0.01
    probs[0, digit_to_index(3)] = 0.60
    probs[0, digit_to_index(5)] = 0.30
    probs[0, digit_to_index(2)] = 0.05

    idx_40 = 4 * 9 + 0
    probs[idx_40, :] = 0.01
    probs[idx_40, digit_to_index(8)] = 0.55
    probs[idx_40, digit_to_index(4)] = 0.35
    probs[idx_40, digit_to_index(2)] = 0.05

    ucs_res = search_corrections(corrupted, probs, algo="ucs", max_corrections=3)
    greedy_res = search_corrections(corrupted, probs, algo="greedy", max_corrections=3)
    astar_res = search_corrections(corrupted, probs, algo="astar", max_corrections=3)

    assert ucs_res.solved and greedy_res.solved and astar_res.solved
    assert ucs_res.path_cost <= greedy_res.path_cost + 1e-9
    assert abs(ucs_res.path_cost - astar_res.path_cost) < 1e-6


def test_node_limit_reported():
    bad = KNOWN_PUZZLE.copy()
    bad[0, 0] = 3
    bad[1, 0] = 1
    bad[4, 0] = 8
    probs = _make_confident_probs(bad, high_p=0.90)
    probs[0, digit_to_index(3)] = 0.50
    probs[0, digit_to_index(2)] = 0.45

    res = search_corrections(bad, probs, algo="ucs", max_corrections=3, node_limit=2)
    assert res.solved is False
    assert res.status == "node_limit"
    assert res.limit_hit == "node_limit"


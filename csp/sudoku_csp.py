"""Sudoku as a Constraint Satisfaction Problem (BT, FC, and AC-3)."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import time

import numpy as np

from data.labels import DIGITS


def _build_neighbors() -> list[tuple[int, ...]]:
    """Precompute the 20 peer cell indices (same row, col, or 3x3 box) for each cell 0..80."""
    neighbors: list[tuple[int, ...]] = []
    for idx in range(81):
        r, c = divmod(idx, 9)
        br, bc = 3 * (r // 3), 3 * (c // 3)
        peers = set()
        for k in range(9):
            peers.add(r * 9 + k)
            peers.add(k * 9 + c)
        for dr in range(3):
            for dc in range(3):
                peers.add((br + dr) * 9 + (bc + dc))
        peers.discard(idx)
        neighbors.append(tuple(sorted(peers)))
    return neighbors


NEIGHBORS: list[tuple[int, ...]] = _build_neighbors()
ALL_ARCS: list[tuple[int, int]] = [
    (xi, xj) for xi in range(81) for xj in NEIGHBORS[xi]
]


@dataclass
class CSPResult:
    """Result and solver counters returned by CSPSolver.solve / count_solutions."""

    solved: bool
    status: str  # "solved", "inconsistent", "no solution", "timeout"
    message: str
    solution: np.ndarray | None
    assignments: int
    backtracks: int
    values_pruned: int
    time_sec: float
    num_solutions: int = 0


class CSPSolver:
    """81-variable Sudoku CSP solver with switchable FC and AC-3."""

    def __init__(self, use_fc: bool = True, use_ac3: bool = True):
        self.use_fc = bool(use_fc)
        self.use_ac3 = bool(use_ac3)

    def _init_domains(self, grid: np.ndarray) -> tuple[list[set[int]], dict[int, int]]:
        """Create initial domains {1..9} for blanks and {val} for givens."""
        domains: list[set[int]] = []
        assigned: dict[int, int] = {}
        flat = grid.reshape(81)
        for idx in range(81):
            val = int(flat[idx])
            if val == 0:
                domains.append(set(DIGITS))
            else:
                if val not in DIGITS:
                    raise ValueError(f"Invalid Sudoku value {val} at cell {idx}")
                domains.append({val})
                assigned[idx] = val
        return domains, assigned

    def _givens_conflict(self, assigned: dict[int, int]) -> bool:
        """Return True if two fixed givens in the same unit have the same digit."""
        for cell, val in assigned.items():
            for nbr in NEIGHBORS[cell]:
                if nbr in assigned and assigned[nbr] == val:
                    return True
        return False

    def _revise(self, domains: list[set[int]], xi: int, xj: int) -> int:
        """Remove values in domains[xi] with no compatible value in domains[xj]."""
        removed = 0
        # For Sudoku all-different, x in domains[xi] is invalid iff domains[xj] == {x}.
        if len(domains[xj]) == 1:
            only_val = next(iter(domains[xj]))
            if only_val in domains[xi]:
                domains[xi].remove(only_val)
                removed = 1
        return removed

    def ac3(self, domains: list[set[int]]) -> tuple[bool, int]:
        """Enforce arc consistency across all Sudoku arcs. Returns (consistent, pruned)."""
        queue = deque(ALL_ARCS)
        pruned = 0
        while queue:
            xi, xj = queue.popleft()
            removed = self._revise(domains, xi, xj)
            if removed > 0:
                pruned += removed
                if len(domains[xi]) == 0:
                    return False, pruned
                for xk in NEIGHBORS[xi]:
                    if xk != xj:
                        queue.append((xk, xi))
        return True, pruned

    def _is_consistent(self, cell: int, val: int, assigned: dict[int, int]) -> bool:
        """Check whether assigning `val` to `cell` conflicts with any assigned peer."""
        for nbr in NEIGHBORS[cell]:
            if assigned.get(nbr) == val:
                return False
        return True

    def _select_unassigned_mrv(
        self, domains: list[set[int]], assigned: dict[int, int]
    ) -> int:
        """Pick unassigned variable with Minimum Remaining Values (smallest domain)."""
        best_cell = -1
        best_size = 10
        for cell in range(81):
            if cell not in assigned:
                size = len(domains[cell])
                if size < best_size:
                    best_size = size
                    best_cell = cell
                    if size <= 1:
                        break
        return best_cell

    def _forward_check(
        self,
        domains: list[set[int]],
        assigned: dict[int, int],
        cell: int,
        val: int,
    ) -> tuple[bool, list[tuple[int, int]], int]:
        """Remove `val` from unassigned peers' domains. Returns (ok, undo_list, pruned)."""
        undo: list[tuple[int, int]] = []
        pruned = 0
        for nbr in NEIGHBORS[cell]:
            if nbr not in assigned and val in domains[nbr]:
                domains[nbr].remove(val)
                undo.append((nbr, val))
                pruned += 1
                if len(domains[nbr]) == 0:
                    return False, undo, pruned
        return True, undo, pruned

    def _restore(self, domains: list[set[int]], undo: list[tuple[int, int]]) -> None:
        """Undo domain removals after backtracking."""
        for cell, val in undo:
            domains[cell].add(val)

    def solve(
        self,
        grid: np.ndarray,
        max_solutions: int = 1,
        timeout_sec: float | None = None,
    ) -> CSPResult:
        """Solve a 9x9 Sudoku grid (or count solutions up to `max_solutions`)."""
        start = time.perf_counter()
        domains, assigned = self._init_domains(grid)

        assignments = 0
        backtracks = 0
        values_pruned = 0

        if self.use_ac3:
            consistent, ac3_pruned = self.ac3(domains)
            values_pruned += ac3_pruned
            if not consistent:
                return CSPResult(
                    solved=False,
                    status="inconsistent",
                    message="no solution: inconsistent puzzle (AC-3 emptied a domain)",
                    solution=None,
                    assignments=0,
                    backtracks=0,
                    values_pruned=values_pruned,
                    time_sec=time.perf_counter() - start,
                    num_solutions=0,
                )
        else:
            if self._givens_conflict(assigned):
                return CSPResult(
                    solved=False,
                    status="no solution",
                    message="no solution: conflicting givens in row, column, or box",
                    solution=None,
                    assignments=0,
                    backtracks=0,
                    values_pruned=0,
                    time_sec=time.perf_counter() - start,
                    num_solutions=0,
                )

        solutions_found = 0
        first_solution: np.ndarray | None = None
        timed_out = False

        def _backtrack() -> bool:
            nonlocal assignments, backtracks, values_pruned, solutions_found, first_solution, timed_out
            if timeout_sec is not None and (time.perf_counter() - start) >= timeout_sec:
                timed_out = True
                return True

            if len(assigned) == 81:
                solutions_found += 1
                if first_solution is None:
                    sol = np.zeros(81, dtype=np.int64)
                    for k, v in assigned.items():
                        sol[k] = v
                    first_solution = sol.reshape(9, 9)
                return solutions_found >= max_solutions

            var = self._select_unassigned_mrv(domains, assigned)
            candidates = sorted(domains[var])

            for val in candidates:
                if not self._is_consistent(var, val, assigned):
                    continue

                assigned[var] = val
                assignments += 1

                if self.use_fc:
                    ok, undo, fc_pruned = self._forward_check(domains, assigned, var, val)
                    values_pruned += fc_pruned
                    solved_branch = ok and _backtrack()
                    self._restore(domains, undo)
                else:
                    solved_branch = _backtrack()

                del assigned[var]
                if solved_branch:
                    return True
                if not timed_out:
                    backtracks += 1

            return False

        _backtrack()
        elapsed = time.perf_counter() - start

        if timed_out and first_solution is None:
            return CSPResult(
                solved=False,
                status="timeout",
                message=f"no solution: solver timed out after {timeout_sec:.2f}s",
                solution=None,
                assignments=assignments,
                backtracks=backtracks,
                values_pruned=values_pruned,
                time_sec=elapsed,
                num_solutions=solutions_found,
            )

        if first_solution is not None:
            return CSPResult(
                solved=True,
                status="solved",
                message="solved",
                solution=first_solution,
                assignments=assignments,
                backtracks=backtracks,
                values_pruned=values_pruned,
                time_sec=elapsed,
                num_solutions=solutions_found,
            )

        return CSPResult(
            solved=False,
            status="no solution",
            message="no solution: search exhausted all assignments",
            solution=None,
            assignments=assignments,
            backtracks=backtracks,
            values_pruned=values_pruned,
            time_sec=elapsed,
            num_solutions=0,
        )


def count_solutions(grid: np.ndarray, limit: int = 2) -> int:
    """Count Sudoku solutions up to `limit` using FC + AC-3 CSP solver."""
    res = CSPSolver(use_fc=True, use_ac3=True).solve(grid, max_solutions=limit)
    return res.num_solutions


def _run_sanity_check(seed: int = 42, num_puzzles: int = 20, num_givens: int = 30) -> None:
    """Solve generated puzzles with BT, BT+FC, and BT+FC+AC-3 and compare stats."""
    from data.sudoku_generator import generate_puzzle

    rng = np.random.default_rng(seed)
    puzzles = [generate_puzzle(num_givens=num_givens, rng=rng)[0] for _ in range(num_puzzles)]
    configs = [
        ("BT", CSPSolver(use_fc=False, use_ac3=False)),
        ("BT+FC", CSPSolver(use_fc=True, use_ac3=False)),
        ("BT+FC+AC-3", CSPSolver(use_fc=True, use_ac3=True)),
    ]

    ref_solutions: list[np.ndarray] = []
    print(f"CSP Sanity Check: {num_puzzles} puzzles ({num_givens} givens, seed={seed})")
    for name, solver in configs:
        assigns, backs, pruned, times = [], [], [], []
        for i, puz in enumerate(puzzles):
            res = solver.solve(puz)
            assert res.solved and res.solution is not None
            if name == "BT":
                ref_solutions.append(res.solution)
            else:
                assert np.array_equal(res.solution, ref_solutions[i])
            assigns.append(res.assignments)
            backs.append(res.backtracks)
            pruned.append(res.values_pruned)
            times.append(res.time_sec * 1000.0)
        print(
            f"  {name:11s} | avg_assignments={np.mean(assigns):8.2f} | "
            f"avg_backtracks={np.mean(backs):8.2f} | "
            f"avg_pruned={np.mean(pruned):7.2f} | "
            f"avg_time={np.mean(times):6.2f} ms"
        )
    print("All 20 solutions are identical across BT, BT+FC, and BT+FC+AC-3.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CSP sanity check on generated puzzles.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--puzzles", type=int, default=20)
    parser.add_argument("--givens", type=int, default=30)
    args = parser.parse_args()
    _run_sanity_check(seed=args.seed, num_puzzles=args.puzzles, num_givens=args.givens)

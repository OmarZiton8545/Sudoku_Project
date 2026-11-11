"""End-to-end handwritten Sudoku reader, correction search, and CSP solver pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from csp.sudoku_csp import CSPResult, CSPSolver
from integration.reader import GridReading, read_grid
from nn.model import DigitMLP
from search.correction_search import (
    DEFAULT_MAX_CORRECTIONS,
    SearchResult,
    find_duplicate_conflicts,
    get_least_confident_cells,
    search_corrections,
)


@dataclass
class PipelineResult:
    """Complete result from solving a 252x252 Sudoku mosaic end to end."""

    solved: bool
    status: str
    message: str
    first_grid: np.ndarray
    corrected_grid: np.ndarray | None
    solution: np.ndarray | None
    probs: np.ndarray
    blank_mask: np.ndarray
    initial_conflicts: list[tuple[tuple[int, int], tuple[int, int], int]]
    initial_csp: CSPResult
    final_csp: CSPResult | None
    search_result: SearchResult | None
    correction_log: list[dict]
    least_confident_cells: list[dict]
    trace: list[dict]
    no_nn: bool


def compare_csp_configs(grid: np.ndarray) -> dict[str, CSPResult]:
    """Run BT, BT+FC, and BT+FC+AC-3 on the same grid for pruning comparison."""
    return {
        "BT": CSPSolver(use_fc=False, use_ac3=False).solve(grid),
        "BT+FC": CSPSolver(use_fc=True, use_ac3=False).solve(grid),
        "BT+FC+AC-3": CSPSolver(use_fc=True, use_ac3=True).solve(grid),
    }


def _describe_conflicts(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    """Return duplicate peer conflicts as [((r1, c1), (r2, c2), digit), ...]."""
    flat = grid.reshape(81)
    pairs = find_duplicate_conflicts(grid)
    return [(divmod(u, 9), divmod(v, 9), int(flat[u])) for u, v in pairs]


def solve_sudoku_mosaic(
    mosaic: np.ndarray,
    model: DigitMLP,
    temperature: float = 1.0,
    search_algo: str = "astar",
    max_corrections: int = DEFAULT_MAX_CORRECTIONS,
    require_unique: bool = False,
    no_nn: bool = False,
    timeout_sec: float | None = 5.0,
    node_limit: int | None = 10_000,
    record_trace: bool = True,
) -> PipelineResult:
    """Read a 252x252 mosaic, check conflicts with AC-3, search corrections, and solve.

    If `no_nn=True`, skips the NN-guided correction search and directly tries to solve
    the raw first-reading grid with the CSP.
    """
    reading: GridReading = read_grid(mosaic, model, temperature=temperature)
    initial_conflicts = _describe_conflicts(reading.first_grid)
    least_conf = get_least_confident_cells(reading.first_grid, reading.probs, top_n=5)

    csp = CSPSolver(use_fc=True, use_ac3=True)
    max_solutions = 2 if require_unique else 1
    initial_csp = csp.solve(reading.first_grid, max_solutions=max_solutions)

    if no_nn:
        solved_direct = initial_csp.solved and (
            not require_unique or initial_csp.num_solutions == 1
        )
        status = "solved" if solved_direct else initial_csp.status
        msg = (
            "solved without correction search (--no-nn)"
            if solved_direct
            else f"unreadable (--no-nn skipped corrections): {initial_csp.message}"
        )
        return PipelineResult(
            solved=solved_direct,
            status=status,
            message=msg,
            first_grid=reading.first_grid,
            corrected_grid=reading.first_grid.copy() if solved_direct else None,
            solution=initial_csp.solution if solved_direct else None,
            probs=reading.probs,
            blank_mask=reading.blank_mask,
            initial_conflicts=initial_conflicts,
            initial_csp=initial_csp,
            final_csp=initial_csp if solved_direct else None,
            search_result=None,
            correction_log=[],
            least_confident_cells=least_conf,
            trace=[],
            no_nn=True,
        )

    search_res = search_corrections(
        first_grid=reading.first_grid,
        probs=reading.probs,
        algo=search_algo,
        max_corrections=max_corrections,
        require_unique=require_unique,
        timeout_sec=timeout_sec,
        node_limit=node_limit,
        record_trace=record_trace,
    )

    final_csp: CSPResult | None = None
    if search_res.solved and search_res.corrected_grid is not None:
        final_csp = csp.solve(search_res.corrected_grid, max_solutions=max_solutions)

    return PipelineResult(
        solved=search_res.solved,
        status=search_res.status,
        message=search_res.message,
        first_grid=reading.first_grid,
        corrected_grid=search_res.corrected_grid,
        solution=search_res.solution,
        probs=reading.probs,
        blank_mask=reading.blank_mask,
        initial_conflicts=initial_conflicts,
        initial_csp=initial_csp,
        final_csp=final_csp,
        search_result=search_res,
        correction_log=search_res.correction_log,
        least_confident_cells=search_res.least_confident_cells,
        trace=search_res.trace,
        no_nn=False,
    )

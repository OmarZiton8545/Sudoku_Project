"""Search over Sudoku cell corrections using UCS, Greedy Best-First, and A*."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import heapq
import math
import time

import numpy as np

from csp.sudoku_csp import CSPSolver, NEIGHBORS
from data.labels import DIGITS, digit_to_index, index_to_digit


DEFAULT_MAX_CORRECTIONS = 3
DEFAULT_TOP_K_ALTS = 3
DEFAULT_LOW_CONF_THRESHOLD = 1.0
DEFAULT_MAX_LOW_CONF_CELLS = 10
PROB_EPS = 1e-12


@dataclass
class SearchResult:
    """Output of a correction search run (UCS, Greedy, or A*)."""

    solved: bool
    status: str  # "solved", "unreadable", "timeout", "node_limit"
    message: str
    solution: np.ndarray | None
    corrected_grid: np.ndarray | None
    correction_log: list[dict]
    least_confident_cells: list[dict]
    nodes_expanded: int
    nodes_generated: int
    goal_checks: int
    path_cost: float
    time_sec: float
    limit_hit: str | None = None
    trace: list[dict] = field(default_factory=list)


def action_cost(p_read: float, p_new: float) -> float:
    """Non-negative log-probability penalty for changing a cell to an alternative digit."""
    cost = math.log(max(float(p_read), PROB_EPS)) - math.log(max(float(p_new), PROB_EPS))
    return max(0.0, cost)


def find_duplicate_conflicts(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return all unique unordered peer cell pairs (u < v) sharing the same non-zero digit."""
    flat = grid.reshape(81)
    conflicts: list[tuple[int, int]] = []
    for u in range(81):
        val = int(flat[u])
        if val == 0:
            continue
        for v in NEIGHBORS[u]:
            if u < v and int(flat[v]) == val:
                conflicts.append((u, v))
    return conflicts


def get_least_confident_cells(
    first_grid: np.ndarray, probs: np.ndarray, top_n: int = 5
) -> list[dict]:
    """Return the `top_n` lowest-confidence non-blank cells."""
    flat = first_grid.reshape(81)
    items = []
    for idx in range(81):
        d = int(flat[idx])
        if d != 0:
            p_read = float(probs[idx, digit_to_index(d)])
            r, c = divmod(idx, 9)
            items.append(
                {
                    "cell": (r, c),
                    "cell_index": idx,
                    "read_digit": d,
                    "confidence": p_read,
                }
            )
    items.sort(key=lambda item: item["confidence"])
    return items[:top_n]


def build_candidate_actions(
    first_grid: np.ndarray,
    probs: np.ndarray,
    top_k_alts: int = DEFAULT_TOP_K_ALTS,
    low_conf_threshold: float = DEFAULT_LOW_CONF_THRESHOLD,
    max_low_conf: int = DEFAULT_MAX_LOW_CONF_CELLS,
) -> dict[int, list[tuple[int, float]]]:
    """Select candidate cells and their top-k alternative digits with costs.

    Candidate cell rule:
      1. Include ALL non-blank cells currently involved in a duplicate-digit conflict.
      2. Also include the lowest-confidence non-blank cells (confidence < low_conf_threshold,
         capped at `max_low_conf` cells) so misreads that break global solvability without an
         immediate duplicate conflict can still be corrected.
    """
    flat = first_grid.reshape(81)
    conflicts = find_duplicate_conflicts(first_grid)
    conflict_cells = {u for u, v in conflicts} | {v for u, v in conflicts}

    non_blank_conf: list[tuple[float, int]] = []
    for idx in range(81):
        d = int(flat[idx])
        if d != 0:
            p_read = float(probs[idx, digit_to_index(d)])
            if p_read < low_conf_threshold:
                non_blank_conf.append((p_read, idx))
    non_blank_conf.sort()
    low_conf_cells = {idx for _, idx in non_blank_conf[:max_low_conf]}

    candidate_cells = sorted(conflict_cells | low_conf_cells)
    actions: dict[int, list[tuple[int, float]]] = {}

    for idx in candidate_cells:
        read_digit = int(flat[idx])
        read_cls = digit_to_index(read_digit)
        p_read = float(probs[idx, read_cls])

        # Rank all alternative digits 1..9 (excluding read_digit) by NN probability descending.
        alts: list[tuple[float, int]] = []
        for d in DIGITS:
            if d == read_digit:
                continue
            p_alt = float(probs[idx, digit_to_index(d)])
            alts.append((p_alt, d))
        alts.sort(key=lambda item: (-item[0], item[1]))

        cell_alts = [(d, action_cost(p_read, p_alt)) for p_alt, d in alts[:top_k_alts]]
        if cell_alts:
            actions[idx] = cell_alts

    return actions


def apply_corrections(
    first_grid: np.ndarray, state: frozenset[tuple[int, int]]
) -> np.ndarray:
    """Return a copy of `first_grid` with all `(cell_idx, new_digit)` corrections applied."""
    flat = first_grid.reshape(81).copy()
    for cell_idx, new_digit in state:
        flat[cell_idx] = new_digit
    return flat.reshape(9, 9)


def conflict_heuristic(
    grid: np.ndarray,
    state: frozenset[tuple[int, int]],
    min_alt_costs: dict[int, float],
) -> float:
    """Admissible lower-bound cost h(state) to resolve all duplicate-digit conflicts.

    Admissibility proof:
      - If `grid` has no duplicate-digit conflicts, returns 0.0.
      - Otherwise, let P be a set of pairwise cell-disjoint duplicate conflict pairs (u, v).
      - Breaking each conflict (u, v) requires changing at least one of u or v.
      - Since pairs in P share no cells, no single cell change can break two pairs in P.
      - Any cell not yet corrected in `state` costs at least `min_alt_costs[cell]` (>= 0)
        to change; cells already corrected in `state` cannot be corrected again (cost inf).
      - Summing min(cost(u), cost(v)) over disjoint pairs in P therefore never overestimates
        the true remaining correction cost to reach a conflict-free (solvable) grid.
    """
    conflicts = find_duplicate_conflicts(grid)
    if not conflicts:
        return 0.0

    corrected_cells = {cell for cell, _ in state}

    def _cell_min_cost(cell: int) -> float:
        if cell in corrected_cells:
            return math.inf
        return min_alt_costs.get(cell, 0.0)

    # Sort pairs by largest min-change cost first for a tighter disjoint matching.
    scored = sorted(
        ((min(_cell_min_cost(u), _cell_min_cost(v)), u, v) for u, v in conflicts),
        key=lambda item: -item[0],
    )
    used: set[int] = set()
    h_val = 0.0
    for pair_cost, u, v in scored:
        if u not in used and v not in used:
            used.add(u)
            used.add(v)
            if math.isinf(pair_cost):
                return math.inf
            h_val += pair_cost
    return h_val


def search_corrections(
    first_grid: np.ndarray,
    probs: np.ndarray,
    algo: str = "astar",
    max_corrections: int = DEFAULT_MAX_CORRECTIONS,
    top_k_alts: int = DEFAULT_TOP_K_ALTS,
    require_unique: bool = False,
    timeout_sec: float | None = 5.0,
    node_limit: int | None = 10_000,
    record_trace: bool = True,
) -> SearchResult:
    """Find a cheapest/plausible set of cell corrections so CSP (BT+FC+AC-3) solves the grid.

    Algorithms:
      - "ucs": priority = g (cumulative path cost)
      - "greedy": priority = (h, last_action_cost) — greedily resolves conflicts with
        cheapest single action, ignoring cumulative path cost g
      - "astar": priority = g + h (admissible conflict-resolution heuristic)
    """
    algo_norm = algo.lower().strip()
    if algo_norm not in ("ucs", "greedy", "astar"):
        raise ValueError(f"Unsupported search algorithm {algo!r}; expected ucs, greedy, or astar")

    start = time.perf_counter()
    csp = CSPSolver(use_fc=True, use_ac3=True)
    max_solutions = 2 if require_unique else 1
    least_conf = get_least_confident_cells(first_grid, probs, top_n=5)

    def _is_goal(grid: np.ndarray) -> tuple[bool, np.ndarray | None]:
        res = csp.solve(grid, max_solutions=max_solutions)
        if not res.solved:
            return False, None
        if require_unique and res.num_solutions != 1:
            return False, None
        return True, res.solution

    # Check if the initial reading is already conflict-free and solvable (0 corrections).
    goal_checks = 1
    solved0, sol0 = _is_goal(first_grid)
    if solved0:
        return SearchResult(
            solved=True,
            status="solved",
            message="solved with 0 corrections",
            solution=sol0,
            corrected_grid=first_grid.copy(),
            correction_log=[],
            least_confident_cells=least_conf,
            nodes_expanded=0,
            nodes_generated=1,
            goal_checks=goal_checks,
            path_cost=0.0,
            time_sec=time.perf_counter() - start,
            limit_hit=None,
            trace=[{"step": 0, "corrections": [], "g": 0.0, "h": 0.0, "f": 0.0, "status": "solved"}],
        )

    actions = build_candidate_actions(first_grid, probs, top_k_alts=top_k_alts)
    min_alt_costs = {
        cell: min(cost for _, cost in alts) for cell, alts in actions.items()
    }

    init_state: frozenset[tuple[int, int]] = frozenset()
    init_h = conflict_heuristic(first_grid, init_state, min_alt_costs)
    init_prio = 0.0 if algo_norm == "ucs" else (init_h if algo_norm == "astar" else (init_h, 0.0))

    # Heap entries: (priority, tie_id, g_cost, state, h_val)
    tie_counter = 0
    frontier: list[tuple] = [(init_prio, tie_counter, 0.0, init_state, init_h)]
    best_g: dict[frozenset[tuple[int, int]], float] = {init_state: 0.0}
    closed: set[frozenset[tuple[int, int]]] = set()

    nodes_expanded = 0
    nodes_generated = 1
    trace: list[dict] = []
    limit_hit: str | None = None

    while frontier:
        if timeout_sec is not None and (time.perf_counter() - start) >= timeout_sec:
            limit_hit = "timeout"
            break
        if node_limit is not None and nodes_expanded >= node_limit:
            limit_hit = "node_limit"
            break

        prio, _, g_cost, state, h_val = heapq.heappop(frontier)
        if state in closed:
            continue
        if g_cost > best_g.get(state, math.inf) + 1e-12:
            continue

        closed.add(state)
        nodes_expanded += 1
        grid = apply_corrections(first_grid, state)

        if record_trace and len(trace) < 200:
            f_val = prio[0] if isinstance(prio, tuple) else prio
            corr_pretty = [
                {
                    "cell": divmod(cell, 9),
                    "read": int(first_grid.reshape(81)[cell]),
                    "new": digit,
                }
                for cell, digit in sorted(state)
            ]
            trace.append(
                {
                    "step": nodes_expanded,
                    "corrections": corr_pretty,
                    "g": round(g_cost, 6),
                    "h": round(h_val, 6),
                    "f": round(float(f_val), 6),
                }
            )

        active_conflicts = find_duplicate_conflicts(grid)
        # Skip CSP goal test on root (already tested) or states that still have duplicate conflicts.
        if len(state) > 0 and not active_conflicts:
            goal_checks += 1
            is_ok, sol = _is_goal(grid)
            if is_ok:
                flat_first = first_grid.reshape(81)
                correction_log = []
                for cell_idx, new_digit in sorted(state):
                    read_digit = int(flat_first[cell_idx])
                    p_read = float(probs[cell_idx, digit_to_index(read_digit)])
                    p_new = float(probs[cell_idx, digit_to_index(new_digit)])
                    r, c = divmod(cell_idx, 9)
                    correction_log.append(
                        {
                            "cell": (r, c),
                            "cell_index": cell_idx,
                            "read_digit": read_digit,
                            "new_digit": new_digit,
                            "cost": action_cost(p_read, p_new),
                        }
                    )
                if trace:
                    trace[-1]["status"] = "solved"
                return SearchResult(
                    solved=True,
                    status="solved",
                    message=f"solved with {len(correction_log)} correction(s)",
                    solution=sol,
                    corrected_grid=grid,
                    correction_log=correction_log,
                    least_confident_cells=least_conf,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    goal_checks=goal_checks,
                    path_cost=g_cost,
                    time_sec=time.perf_counter() - start,
                    limit_hit=None,
                    trace=trace,
                )

        if len(state) >= max_corrections:
            continue

        corrected_cells = {cell for cell, _ in state}
        # If duplicate conflicts remain and only 1 correction budget remains,
        # any valid fix must touch a cell in an active conflict.
        conflict_set = {u for u, v in active_conflicts} | {v for u, v in active_conflicts}
        if len(state) == max_corrections - 1 and conflict_set:
            expand_cells = [c for c in sorted(conflict_set) if c in actions and c not in corrected_cells]
        else:
            expand_cells = [c for c in actions if c not in corrected_cells]

        for cell_idx in expand_cells:
            for new_digit, step_cost in actions[cell_idx]:
                next_state = state | {(cell_idx, new_digit)}
                if next_state in closed:
                    continue
                next_g = g_cost + step_cost
                if next_g >= best_g.get(next_state, math.inf) - 1e-12:
                    continue

                next_grid = apply_corrections(first_grid, next_state)
                next_h = conflict_heuristic(next_grid, next_state, min_alt_costs)
                if math.isinf(next_h):
                    continue

                best_g[next_state] = next_g
                nodes_generated += 1
                tie_counter += 1

                if algo_norm == "ucs":
                    next_prio: float | tuple[float, float] = next_g
                elif algo_norm == "astar":
                    next_prio = next_g + next_h
                else:
                    next_prio = (next_h, step_cost)

                heapq.heappush(frontier, (next_prio, tie_counter, next_g, next_state, next_h))

    elapsed = time.perf_counter() - start
    if limit_hit == "timeout":
        status = "timeout"
        msg = f"unreadable: search timed out after {elapsed:.2f}s (least confident: {least_conf[:3]})"
    elif limit_hit == "node_limit":
        status = "node_limit"
        msg = f"unreadable: node limit ({node_limit}) reached (least confident: {least_conf[:3]})"
    else:
        status = "unreadable"
        msg = (
            f"unreadable: no solvable grid within {max_corrections} corrections "
            f"(least confident: {least_conf[:3]})"
        )

    return SearchResult(
        solved=False,
        status=status,
        message=msg,
        solution=None,
        corrected_grid=None,
        correction_log=[],
        least_confident_cells=least_conf,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        goal_checks=goal_checks,
        path_cost=math.inf,
        time_sec=elapsed,
        limit_hit=limit_hit,
        trace=trace,
    )


def _run_quick_check(seed: int = 42, num_puzzles: int = 20, num_givens: int = 30) -> None:
    """Run UCS, Greedy, and A* on 20 generated mosaics for k=1, 2, 3 misreads."""
    from data.mnist_loader import load_processed_mnist
    from data.sudoku_generator import build_hard_digit_pools, generate_puzzle
    from integration.reader import build_sudoku_mosaic, read_grid
    from nn.model import predict_probs
    from nn.train import load_trained_model

    data = load_processed_mnist("data/processed")
    model, temperature = load_trained_model("nn/checkpoints/digit_mlp.pt")
    test_probs = predict_probs(model, data["x_test"], temperature=temperature)
    pools = build_hard_digit_pools(data["y_test"], test_probs)

    print(f"Search Quick Check ({num_puzzles} mosaics/level, {num_givens} givens, seed={seed}):")
    for k in (1, 2, 3):
        rng = np.random.default_rng(seed + 1000 * k)
        samples = []
        for _ in range(num_puzzles):
            puzzle, solution = generate_puzzle(num_givens=num_givens, rng=rng)
            mosaic, _ = build_sudoku_mosaic(
                puzzle=puzzle, test_images=data["x_test"], pools=pools, num_misreads=k, rng=rng
            )
            reading = read_grid(mosaic, model, temperature=temperature)
            samples.append((reading, solution))

        print(f"  --- k={k} misread(s) ---")
        for algo in ("ucs", "greedy", "astar"):
            exact_rec = 0
            nodes_exp = []
            timeouts = 0
            for reading, solution in samples:
                res = search_corrections(
                    reading.first_grid,
                    reading.probs,
                    algo=algo,
                    max_corrections=3,
                    timeout_sec=5.0,
                    record_trace=False,
                )
                if res.solved and res.solution is not None and np.array_equal(res.solution, solution):
                    exact_rec += 1
                if res.limit_hit is not None:
                    timeouts += 1
                nodes_exp.append(res.nodes_expanded)
            print(
                f"    {algo:6s} | recovery={exact_rec}/{num_puzzles} ({exact_rec/num_puzzles:.1%}) | "
                f"avg_nodes_expanded={np.mean(nodes_exp):7.2f} | limits_hit={timeouts}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick check of UCS, Greedy, and A* on mosaics.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--puzzles", type=int, default=20)
    parser.add_argument("--givens", type=int, default=30)
    args = parser.parse_args()
    _run_quick_check(seed=args.seed, num_puzzles=args.puzzles, num_givens=args.givens)

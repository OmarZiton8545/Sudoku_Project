"""Diagnostic script comparing UCS (require_unique=False/True) and A* path cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.mnist_loader import load_processed_mnist
from data.sudoku_generator import build_hard_digit_pools, generate_puzzle
from integration.reader import build_sudoku_mosaic, read_grid
from nn.model import predict_probs
from nn.train import load_trained_model
from search.correction_search import search_corrections


def _classify_failure(res, true_solution: np.ndarray) -> str:
    """Classify a non-recovered puzzle into (a), (b), or (c)."""
    if not res.solved:
        return "c_budget_or_unreadable"
    if len(res.correction_log) == 0:
        return "a_no_conflict_0_corrections"
    return "b_different_cheaper_fix"


def run_diagnostic(
    seed: int = 42,
    num_puzzles: int = 20,
    num_givens: int = 30,
    out_path: str | Path = "results/search_diagnostic.json",
) -> dict:
    """Compare UCS(require_unique=False/True) and verify UCS==A* path cost."""
    data = load_processed_mnist("data/processed")
    model, temperature = load_trained_model("nn/checkpoints/digit_mlp.pt")
    test_probs = predict_probs(model, data["x_test"], temperature=temperature)
    pools = build_hard_digit_pools(data["y_test"], test_probs)

    results = {}
    cost_mismatches = []

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

        level_stats = {}
        for req_unique in (False, True):
            key = f"ucs_unique_{req_unique}"
            recovered = 0
            nodes = []
            failures = {
                "a_no_conflict_0_corrections": 0,
                "b_different_cheaper_fix": 0,
                "c_budget_or_unreadable": 0,
            }

            for idx, (reading, solution) in enumerate(samples):
                ucs_res = search_corrections(
                    reading.first_grid,
                    reading.probs,
                    algo="ucs",
                    max_corrections=3,
                    require_unique=req_unique,
                    timeout_sec=5.0,
                    record_trace=False,
                )
                astar_res = search_corrections(
                    reading.first_grid,
                    reading.probs,
                    algo="astar",
                    max_corrections=3,
                    require_unique=req_unique,
                    timeout_sec=5.0,
                    record_trace=False,
                )

                # Verify UCS and A* return the exact same path cost on every puzzle.
                if ucs_res.solved != astar_res.solved or (
                    ucs_res.solved and abs(ucs_res.path_cost - astar_res.path_cost) > 1e-6
                ):
                    cost_mismatches.append(
                        {
                            "k": k,
                            "puzzle_idx": idx,
                            "require_unique": req_unique,
                            "ucs_solved": ucs_res.solved,
                            "astar_solved": astar_res.solved,
                            "ucs_cost": None if not ucs_res.solved else ucs_res.path_cost,
                            "astar_cost": None if not astar_res.solved else astar_res.path_cost,
                        }
                    )

                nodes.append(ucs_res.nodes_expanded)
                if ucs_res.solved and ucs_res.solution is not None and np.array_equal(ucs_res.solution, solution):
                    recovered += 1
                else:
                    failures[_classify_failure(ucs_res, solution)] += 1

            level_stats[key] = {
                "require_unique": req_unique,
                "num_puzzles": num_puzzles,
                "recovered": recovered,
                "recovery_rate": recovered / num_puzzles,
                "avg_nodes_expanded": float(np.mean(nodes)),
                "failure_breakdown": failures,
            }
        results[f"k={k}"] = level_stats

    assert len(cost_mismatches) == 0, f"UCS vs A* mismatches found: {cost_mismatches}"

    payload = {
        "seed": seed,
        "num_puzzles": num_puzzles,
        "num_givens": num_givens,
        "ucs_vs_astar_cost_mismatches": cost_mismatches,
        "levels": results,
    }
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic on UCS require_unique and UCS==A* cost.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--puzzles", type=int, default=20)
    parser.add_argument("--givens", type=int, default=30)
    args = parser.parse_args()

    payload = run_diagnostic(seed=args.seed, num_puzzles=args.puzzles, num_givens=args.givens)
    print(f"Search Diagnostic (seed={args.seed}, puzzles={args.puzzles}, givens={args.givens}):")
    print(f"UCS vs A* cost mismatches: {len(payload['ucs_vs_astar_cost_mismatches'])}")
    for lvl, data in payload["levels"].items():
        print(f"  {lvl}:")
        for key, st in data.items():
            print(
                f"    {key:17s} | recovery={st['recovered']}/{st['num_puzzles']} ({st['recovery_rate']:.1%}) | "
                f"avg_nodes={st['avg_nodes_expanded']:6.2f} | failures={st['failure_breakdown']}"
            )


if __name__ == "__main__":
    main()


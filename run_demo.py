"""Integrated demo for Project P7: Handwritten Sudoku Reader and Solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.mnist_loader import load_processed_mnist
from data.sudoku_generator import build_hard_digit_pools, generate_puzzle
from integration.pipeline import compare_csp_configs, solve_sudoku_mosaic
from integration.reader import build_sudoku_mosaic
from nn.model import predict_probs
from nn.train import load_trained_model
from search.correction_search import DEFAULT_MAX_CORRECTIONS


def _format_grid(grid: np.ndarray) -> str:
    """Format a 9x9 Sudoku grid for terminal output."""
    lines = []
    for r in range(9):
        if r in (3, 6):
            lines.append("------+-------+------")
        row_tokens = []
        for c in range(9):
            if c in (3, 6):
                row_tokens.append("|")
            val = int(grid[r, c])
            row_tokens.append(str(val) if val != 0 else ".")
        lines.append(" ".join(row_tokens))
    return "\n".join(lines)


def _save_demo_plots(
    mosaic: np.ndarray,
    first_grid: np.ndarray,
    probs: np.ndarray,
    correction_log: list[dict],
    csp_stats: dict,
    algo_stats: dict,
    demo_dir: Path,
) -> dict[str, str]:
    """Save the 4 required demo plots into results/demo/."""
    demo_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # 1. Input 252x252 mosaic.
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(mosaic, cmap="gray")
    ax.set_title("Input 252x252 Handwritten Sudoku Mosaic")
    ax.set_xticks(np.arange(0, 253, 28))
    ax.set_yticks(np.arange(0, 253, 28))
    ax.grid(color="cyan", alpha=0.25, linewidth=0.6)
    ax.tick_params(labelbottom=False, labelleft=False)
    fig.tight_layout()
    p1 = demo_dir / "input_mosaic.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths["input_mosaic"] = str(p1)

    # 2. Grid with corrected cells highlighted.
    fig, ax = plt.subplots(figsize=(6, 5.5))
    conf_map = np.ones((9, 9), dtype=np.float32)
    for idx in range(81):
        r, c = divmod(idx, 9)
        if first_grid[r, c] != 0:
            conf_map[r, c] = float(np.max(probs[idx]))
    im = ax.imshow(conf_map, cmap="YlGn", vmin=0.3, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NN confidence")
    corrected_map = {tuple(e["cell"]): e for e in correction_log}
    for r in range(9):
        for c in range(9):
            d = int(first_grid[r, c])
            if d == 0:
                continue
            if (r, c) in corrected_map:
                new_d = corrected_map[(r, c)]["new_digit"]
                rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.2)
                ax.add_patch(rect)
                ax.text(
                    c, r, f"{d}->{new_d}",
                    ha="center", va="center", color="red", fontsize=8, fontweight="bold"
                )
            else:
                ax.text(c, r, str(d), ha="center", va="center", color="black", fontsize=9)
    ax.set_title("First-Reading Confidence & Corrected Cells")
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    fig.tight_layout()
    p2 = demo_dir / "corrected_grid.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths["corrected_grid"] = str(p2)

    # 3. Bar chart of CSP backtracks (BT vs BT+FC vs BT+FC+AC-3) on this puzzle.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    names = list(csp_stats.keys())
    backtracks = [csp_stats[n].backtracks for n in names]
    bars = ax.bar(names, backtracks, color=["#d95f02", "#7570b3", "#1b9e77"])
    for bar, val in zip(bars, backtracks):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, str(val), ha="center", va="bottom", fontsize=9)
    if max(backtracks) > 20:
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
    ax.set_ylabel("Backtracks")
    ax.set_title("CSP Backtracks on Target Puzzle (BT vs BT+FC vs BT+FC+AC-3)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p3 = demo_dir / "csp_backtracks.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths["csp_backtracks"] = str(p3)

    # 4. Bar chart of search nodes (UCS vs Greedy vs A*).
    fig, ax = plt.subplots(figsize=(6, 4.5))
    algos = list(algo_stats.keys())
    nodes = [algo_stats[a]["nodes_expanded"] for a in algos]
    bars = ax.bar(algos, nodes, color=["#386cb0", "#f0027f", "#66a61e"])
    for bar, val in zip(bars, nodes):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, str(val), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Nodes Expanded")
    ax.set_title("Search Nodes Expanded (UCS vs Greedy vs A*)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p4 = demo_dir / "search_nodes.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    paths["search_nodes"] = str(p4)

    return paths


def run_demo(
    seed: int = 42,
    misreads: int = 2,
    givens: int = 30,
    algo: str = "astar",
    max_corrections: int | None = None,
    require_unique: bool = False,
    no_nn: bool = False,
    show_trace: bool = False,
    timeout_sec: float = 5.0,
) -> dict:
    """Run the integrated Sudoku Reader & Solver demo and save plots to results/demo/."""
    if max_corrections is None:
        max_corrections = 4 if misreads >= 4 else DEFAULT_MAX_CORRECTIONS

    data = load_processed_mnist("data/processed")
    model, temperature = load_trained_model("nn/checkpoints/digit_mlp.pt")
    test_probs = predict_probs(model, data["x_test"], temperature=temperature)
    pools = build_hard_digit_pools(data["y_test"], test_probs)

    rng = np.random.default_rng(seed)
    puzzle, true_solution = generate_puzzle(num_givens=givens, rng=rng)
    mosaic, injected_cells = build_sudoku_mosaic(
        puzzle=puzzle, test_images=data["x_test"], pools=pools, num_misreads=misreads, rng=rng
    )

    res = solve_sudoku_mosaic(
        mosaic=mosaic,
        model=model,
        temperature=temperature,
        search_algo=algo,
        max_corrections=max_corrections,
        require_unique=require_unique,
        no_nn=no_nn,
        timeout_sec=timeout_sec,
        record_trace=True,
    )

    # Compare UCS, Greedy, and A* on the same mosaic for the search nodes bar chart.
    algo_stats = {}
    for a in ("ucs", "greedy", "astar"):
        r_a = solve_sudoku_mosaic(
            mosaic=mosaic,
            model=model,
            temperature=temperature,
            search_algo=a,
            max_corrections=max_corrections,
            require_unique=require_unique,
            no_nn=False,
            timeout_sec=timeout_sec,
            record_trace=False,
        )
        algo_stats[a.upper()] = {
            "solved": r_a.solved,
            "nodes_expanded": r_a.search_result.nodes_expanded if r_a.search_result else 0,
            "path_cost": r_a.search_result.path_cost if r_a.search_result and r_a.solved else None,
            "exact_recovery": bool(
                r_a.solved and r_a.solution is not None and np.array_equal(r_a.solution, true_solution)
            ),
        }

    csp_target = res.corrected_grid if res.corrected_grid is not None else puzzle
    csp_stats = compare_csp_configs(csp_target)

    print(
        f"=== Handwritten Sudoku Demo (seed={seed}, givens={givens}, "
        f"misreads={misreads}, algo={algo}, no_nn={no_nn}) ==="
    )
    print(f"Step 1 - Generated puzzle ({givens} givens) & 252x252 mosaic. Injected misread cells: {injected_cells}")
    print("\nStep 2 - NN First-Reading Grid:")
    print(_format_grid(res.first_grid))

    print(
        f"\nStep 3 - AC-3 Initial Check: status={res.initial_csp.status} | "
        f"message='{res.initial_csp.message}' | duplicate_conflicts={res.initial_conflicts}"
    )

    if no_nn:
        print("\nStep 4 - Search skipped (--no-nn): solving first reading directly with CSP.")
    else:
        nodes_exp = res.search_result.nodes_expanded if res.search_result else 0
        print(f"\nStep 4 - Search ({algo.upper()}): nodes_expanded={nodes_exp}, status={res.status}")
        if res.correction_log:
            print("Correction Log:")
            for entry in res.correction_log:
                print(
                    f"  cell={entry['cell']} | read={entry['read_digit']} -> "
                    f"new={entry['new_digit']} | cost={entry['cost']:.4f}"
                )
        else:
            print("Correction Log: (none)")

        if show_trace:
            print("\nSearch Trace:")
            for step in res.trace:
                print(f"  {step}")

    print("\nStep 5 - Final CSP Solve & Verification:")
    if res.solved and res.solution is not None:
        exact = bool(np.array_equal(res.solution, true_solution))
        print(f"Solved! Equals true solution: {exact}")
        print(_format_grid(res.solution))
    else:
        print(f"unreadable: {res.message}")
        print("Least confident cells:")
        for lc in res.least_confident_cells:
            print(f"  cell={lc['cell']} | read={lc['read_digit']} | confidence={lc['confidence']:.4f}")

    print("\nCSP Pruning Comparison (BT vs BT+FC vs BT+FC+AC-3):")
    for cname, cres in csp_stats.items():
        print(
            f"  {cname:11s} | assignments={cres.assignments:5d} | backtracks={cres.backtracks:5d} | "
            f"pruned={cres.values_pruned:5d} | time={cres.time_sec * 1000:.2f} ms"
        )

    demo_dir = Path("results/demo")
    plot_paths = _save_demo_plots(
        mosaic=mosaic,
        first_grid=res.first_grid,
        probs=res.probs,
        correction_log=res.correction_log,
        csp_stats=csp_stats,
        algo_stats=algo_stats,
        demo_dir=demo_dir,
    )

    summary = {
        "seed": seed,
        "givens": givens,
        "misreads": misreads,
        "algo": algo,
        "no_nn": no_nn,
        "injected_cells": injected_cells,
        "initial_conflicts": res.initial_conflicts,
        "initial_csp_status": res.initial_csp.status,
        "status": res.status,
        "solved": res.solved,
        "exact_recovery": bool(
            res.solved and res.solution is not None and np.array_equal(res.solution, true_solution)
        ),
        "nodes_expanded": res.search_result.nodes_expanded if res.search_result else 0,
        "correction_log": res.correction_log,
        "least_confident_cells": res.least_confident_cells,
        "csp_comparison": {
            k: {
                "assignments": v.assignments,
                "backtracks": v.backtracks,
                "values_pruned": v.values_pruned,
                "time_ms": v.time_sec * 1000.0,
            }
            for k, v in csp_stats.items()
        },
        "algo_comparison": algo_stats,
        "plots": plot_paths,
    }
    (demo_dir / "demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run integrated Handwritten Sudoku Reader & Solver demo.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--misreads", type=int, default=2)
    parser.add_argument("--givens", type=int, default=30)
    parser.add_argument("--algo", type=str, default="astar", choices=("ucs", "greedy", "astar"))
    parser.add_argument("--max-corrections", type=int, default=None)
    parser.add_argument("--require-unique", action="store_true")
    parser.add_argument("--no-nn", "--nn-off", dest="no_nn", action="store_true", help="Skip correction search and solve first reading directly.")
    parser.add_argument("--trace", action="store_true", help="Print step-by-step search trace.")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    run_demo(
        seed=args.seed,
        misreads=args.misreads,
        givens=args.givens,
        algo=args.algo,
        max_corrections=args.max_corrections,
        require_unique=args.require_unique,
        no_nn=args.no_nn,
        show_trace=args.trace,
        timeout_sec=args.timeout,
    )


if __name__ == "__main__":
    main()

"""Run Experiments 1, 2, and 3 across multiple seeds and export CSV, Markdown, and PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csp.sudoku_csp import CSPSolver
from data.labels import DIGITS, digit_to_index
from data.mnist_loader import load_processed_mnist
from data.sudoku_generator import build_hard_digit_pools, generate_puzzle
from integration.reader import build_sudoku_mosaic, read_grid
from nn.model import predict_probs
from nn.train import load_trained_model
from search.correction_search import search_corrections


DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def _make_random_probs(first_grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Create random confidence & alternative rankings for the random-correction baseline."""
    probs = np.zeros((81, 9), dtype=np.float32)
    flat = first_grid.reshape(81)
    for idx in range(81):
        d = int(flat[idx])
        if d == 0:
            continue
        read_cls = digit_to_index(d)
        # Random top-1 confidence in [0.40, 0.90] so random cells look uncertain.
        p_top = float(rng.uniform(0.40, 0.90))
        probs[idx, read_cls] = p_top
        others = [x for x in DIGITS if x != d]
        weights = rng.dirichlet(np.ones(len(others))) * (1.0 - p_top)
        for alt_d, w in zip(others, weights):
            probs[idx, digit_to_index(alt_d)] = float(w)
    return probs


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Format a DataFrame as a GitHub-flavored Markdown table without extra deps."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _format_mean_std(df_seed: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> tuple[pd.DataFrame, str]:
    """Aggregate per-seed metrics into mean, std, and 'mean +/- std' markdown table."""
    grouped = df_seed.groupby(group_cols, as_index=False, sort=False)
    rows = []
    md_rows = []
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        md_row = dict(zip(group_cols, keys))
        for col in metric_cols:
            m = float(sub[col].mean())
            s = float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0
            row[f"{col}_mean"] = m
            row[f"{col}_std"] = s
            if "rate" in col:
                md_row[col] = f"{m * 100:.1f}% +/- {s * 100:.1f}%"
            elif col == "timeouts":
                md_row[col] = f"{int(sub[col].sum())} total ({m:.2f} +/- {s:.2f}/seed)"
            else:
                md_row[col] = f"{m:.2f} +/- {s:.2f}"
        rows.append(row)
        md_rows.append(md_row)

    summary_df = pd.DataFrame(rows)
    md_df = pd.DataFrame(md_rows)
    return summary_df, _df_to_markdown(md_df)


def _prepare_dataset_and_pools():
    """Load processed MNIST, trained DigitMLP, and hard-digit pools."""
    data = load_processed_mnist("data/processed")
    model, temperature = load_trained_model("nn/checkpoints/digit_mlp.pt")
    test_probs = predict_probs(model, data["x_test"], temperature=temperature)
    pools = build_hard_digit_pools(data["y_test"], test_probs)
    return data, model, temperature, pools


def run_exp1(
    seeds: list[int],
    puzzles_per_level: int,
    num_givens: int,
    timeout_sec: float,
    node_limit: int,
    results_dir: Path,
    shared_ctx=None,
) -> tuple[pd.DataFrame, str, float]:
    """Exp 1: UCS vs Greedy vs A* for k=0..4 misreads and require_unique=False/True."""
    t0 = time.perf_counter()
    data, model, temperature, pools = shared_ctx or _prepare_dataset_and_pools()
    raw_rows = []

    for seed in seeds:
        for k in (0, 1, 2, 3, 4):
            max_corr = 4 if k == 4 else 3
            rng = np.random.default_rng(seed + 1000 * k)
            samples = []
            for p_idx in range(puzzles_per_level):
                puzzle, sol = generate_puzzle(num_givens=num_givens, rng=rng)
                mosaic, _ = build_sudoku_mosaic(
                    puzzle=puzzle, test_images=data["x_test"], pools=pools, num_misreads=k, rng=rng
                )
                reading = read_grid(mosaic, model, temperature=temperature)
                samples.append((p_idx, reading, sol))

            for req_unique in (False, True):
                for algo in ("ucs", "greedy", "astar"):
                    for p_idx, reading, sol in samples:
                        res = search_corrections(
                            reading.first_grid,
                            reading.probs,
                            algo=algo,
                            max_corrections=max_corr,
                            require_unique=req_unique,
                            timeout_sec=timeout_sec,
                            node_limit=node_limit,
                            record_trace=False,
                        )
                        recovered = int(
                            res.solved and res.solution is not None and np.array_equal(res.solution, sol)
                        )
                        timed_out = int(res.limit_hit == "timeout")
                        node_lim = int(res.limit_hit == "node_limit")
                        raw_rows.append(
                            {
                                "seed": seed,
                                "k": k,
                                "max_corrections": max_corr,
                                "require_unique": req_unique,
                                "algo": algo.upper(),
                                "puzzle_idx": p_idx,
                                "solved": int(res.solved),
                                "recovery_rate": recovered,
                                "nodes_expanded": res.nodes_expanded,
                                "path_cost": res.path_cost if res.solved else np.nan,
                                "timeouts": timed_out,
                                "node_limits_hit": node_lim,
                                "time_ms": res.time_sec * 1000.0,
                            }
                        )
        print(f"  [Exp 1] finished seed={seed}")

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(results_dir / "exp1_raw.csv", index=False)

    seed_df = raw_df.groupby(["require_unique", "k", "algo", "seed"], as_index=False, sort=False).agg(
        recovery_rate=("recovery_rate", "mean"),
        nodes_expanded=("nodes_expanded", "mean"),
        path_cost=("path_cost", "mean"),
        timeouts=("timeouts", "sum"),
    )
    summary_df, md_table = _format_mean_std(
        seed_df,
        group_cols=["require_unique", "k", "algo"],
        metric_cols=["recovery_rate", "nodes_expanded", "path_cost", "timeouts"],
    )
    summary_df.to_csv(results_dir / "exp1_summary.csv", index=False)
    (results_dir / "exp1_summary.md").write_text(
        f"# Experiment 1: UCS vs Greedy vs A* (k=0..4, {len(seeds)} seeds, {puzzles_per_level} puzzles/level)\n\n"
        + md_table
        + "\n",
        encoding="utf-8",
    )

    # Plot 2x2 grid: top require_unique=False, bottom require_unique=True.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    algo_styles = [("UCS", "#386cb0", "o"), ("GREEDY", "#f0027f", "s"), ("ASTAR", "#66a61e", "^")]
    for row_idx, req_u in enumerate((False, True)):
        sub = summary_df[summary_df["require_unique"] == req_u]
        ax_rec, ax_nodes = axes[row_idx, 0], axes[row_idx, 1]
        for algo, color, marker in algo_styles:
            s_a = sub[sub["algo"] == algo]
            ks = s_a["k"].to_numpy()
            ax_rec.errorbar(
                ks, s_a["recovery_rate_mean"] * 100.0, yerr=s_a["recovery_rate_std"] * 100.0,
                label=algo, color=color, marker=marker, capsize=3, linewidth=1.8
            )
            ax_nodes.errorbar(
                ks, s_a["nodes_expanded_mean"], yerr=s_a["nodes_expanded_std"],
                label=algo, color=color, marker=marker, capsize=3, linewidth=1.8
            )
        ax_rec.set_title(f"Recovery Rate (require_unique={req_u})")
        ax_rec.set_xlabel("Injected misreads (k)")
        ax_rec.set_ylabel("Exact recovery (%)")
        ax_rec.set_ylim(0, 105)
        ax_rec.set_xticks([0, 1, 2, 3, 4])
        ax_rec.grid(True, alpha=0.3)
        ax_rec.legend()

        ax_nodes.set_title(f"Nodes Expanded (require_unique={req_u})")
        ax_nodes.set_xlabel("Injected misreads (k)")
        ax_nodes.set_ylabel("Mean nodes expanded")
        ax_nodes.set_xticks([0, 1, 2, 3, 4])
        ax_nodes.grid(True, alpha=0.3)
        ax_nodes.legend()

    fig.tight_layout()
    fig.savefig(results_dir / "exp1_search.png", dpi=150)
    plt.close(fig)

    elapsed = time.perf_counter() - t0
    return summary_df, md_table, elapsed


def run_exp2(
    seeds: list[int],
    puzzles_per_level: int,
    timeout_sec: float,
    results_dir: Path,
) -> tuple[pd.DataFrame, str, float]:
    """Exp 2: BT vs BT+FC vs BT+FC+AC-3 on 25, 30, 35 givens."""
    t0 = time.perf_counter()
    raw_rows = []
    configs = [
        ("BT", CSPSolver(use_fc=False, use_ac3=False)),
        ("BT+FC", CSPSolver(use_fc=True, use_ac3=False)),
        ("BT+FC+AC-3", CSPSolver(use_fc=True, use_ac3=True)),
    ]

    for seed in seeds:
        for givens in (25, 30, 35):
            rng = np.random.default_rng(seed + 100 * givens)
            puzzles = [generate_puzzle(num_givens=givens, rng=rng)[0] for _ in range(puzzles_per_level)]
            for cname, solver in configs:
                for p_idx, puz in enumerate(puzzles):
                    res = solver.solve(puz, timeout_sec=timeout_sec)
                    raw_rows.append(
                        {
                            "seed": seed,
                            "givens": givens,
                            "solver": cname,
                            "puzzle_idx": p_idx,
                            "solved": int(res.solved),
                            "timeouts": int(res.status == "timeout"),
                            "assignments": res.assignments,
                            "backtracks": res.backtracks,
                            "values_pruned": res.values_pruned,
                            "time_ms": res.time_sec * 1000.0,
                        }
                    )
        print(f"  [Exp 2] finished seed={seed}")

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(results_dir / "exp2_raw.csv", index=False)

    seed_df = raw_df.groupby(["givens", "solver", "seed"], as_index=False, sort=False).agg(
        assignments=("assignments", "mean"),
        backtracks=("backtracks", "mean"),
        time_ms=("time_ms", "mean"),
        timeouts=("timeouts", "sum"),
    )
    summary_df, md_table = _format_mean_std(
        seed_df,
        group_cols=["givens", "solver"],
        metric_cols=["assignments", "backtracks", "time_ms", "timeouts"],
    )
    summary_df.to_csv(results_dir / "exp2_summary.csv", index=False)
    (results_dir / "exp2_summary.md").write_text(
        f"# Experiment 2: BT vs BT+FC vs BT+FC+AC-3 ({len(seeds)} seeds, {puzzles_per_level} puzzles/givens)\n\n"
        + md_table
        + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    styles = [("BT", "#d95f02", "o"), ("BT+FC", "#7570b3", "s"), ("BT+FC+AC-3", "#1b9e77", "^")]
    for cname, color, marker in styles:
        sub = summary_df[summary_df["solver"] == cname]
        gs = sub["givens"].to_numpy()
        axes[0].errorbar(
            gs, sub["backtracks_mean"], yerr=sub["backtracks_std"],
            label=cname, color=color, marker=marker, capsize=3, linewidth=1.8
        )
        axes[1].errorbar(
            gs, sub["time_ms_mean"], yerr=sub["time_ms_std"],
            label=cname, color=color, marker=marker, capsize=3, linewidth=1.8
        )

    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("CSP Backtracks vs Number of Givens")
    axes[0].set_xlabel("Givens")
    axes[0].set_ylabel("Mean backtracks (symlog)")
    axes[0].set_xticks([25, 30, 35])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("CSP Solve Time vs Number of Givens")
    axes[1].set_xlabel("Givens")
    axes[1].set_ylabel("Mean time (ms, symlog)")
    axes[1].set_xticks([25, 30, 35])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(results_dir / "exp2_csp.png", dpi=150)
    plt.close(fig)

    elapsed = time.perf_counter() - t0
    return summary_df, md_table, elapsed


def run_exp3(
    seeds: list[int],
    puzzles_per_level: int,
    num_givens: int,
    timeout_sec: float,
    node_limit: int,
    results_dir: Path,
    require_unique: bool = True,
    shared_ctx=None,
) -> tuple[pd.DataFrame, str, float]:
    """Exp 3: NN-guided vs No correction vs Random correction for k=0..4."""
    t0 = time.perf_counter()
    data, model, temperature, pools = shared_ctx or _prepare_dataset_and_pools()
    csp = CSPSolver(use_fc=True, use_ac3=True)
    max_solutions = 2 if require_unique else 1
    raw_rows = []

    for seed in seeds:
        for k in (0, 1, 2, 3, 4):
            max_corr = 4 if k == 4 else 3
            rng = np.random.default_rng(seed + 1000 * k)
            rand_rng = np.random.default_rng(seed + 9000 * (k + 1))

            for p_idx in range(puzzles_per_level):
                puzzle, sol = generate_puzzle(num_givens=num_givens, rng=rng)
                mosaic, _ = build_sudoku_mosaic(
                    puzzle=puzzle, test_images=data["x_test"], pools=pools, num_misreads=k, rng=rng
                )
                reading = read_grid(mosaic, model, temperature=temperature)

                # 1) NN-guided correction (A*).
                nn_res = search_corrections(
                    reading.first_grid,
                    reading.probs,
                    algo="astar",
                    max_corrections=max_corr,
                    require_unique=require_unique,
                    timeout_sec=timeout_sec,
                    node_limit=node_limit,
                    record_trace=False,
                )
                nn_rec = int(
                    nn_res.solved and nn_res.solution is not None and np.array_equal(nn_res.solution, sol)
                )
                raw_rows.append(
                    {
                        "seed": seed,
                        "k": k,
                        "method": "NN-guided",
                        "puzzle_idx": p_idx,
                        "recovery_rate": nn_rec,
                        "timeouts": int(nn_res.limit_hit == "timeout"),
                        "nodes_expanded": nn_res.nodes_expanded,
                        "time_ms": nn_res.time_sec * 1000.0,
                    }
                )

                # 2) No correction (direct CSP on first reading).
                nc_res = csp.solve(reading.first_grid, max_solutions=max_solutions, timeout_sec=timeout_sec)
                nc_ok = nc_res.solved and (not require_unique or nc_res.num_solutions == 1)
                nc_rec = int(nc_ok and nc_res.solution is not None and np.array_equal(nc_res.solution, sol))
                raw_rows.append(
                    {
                        "seed": seed,
                        "k": k,
                        "method": "No-correction",
                        "puzzle_idx": p_idx,
                        "recovery_rate": nc_rec,
                        "timeouts": int(nc_res.status == "timeout"),
                        "nodes_expanded": 0,
                        "time_ms": nc_res.time_sec * 1000.0,
                    }
                )

                # 3) Random correction (random cell confidences & digit rankings, same budget).
                rand_probs = _make_random_probs(reading.first_grid, rand_rng)
                rc_res = search_corrections(
                    reading.first_grid,
                    rand_probs,
                    algo="ucs",
                    max_corrections=max_corr,
                    require_unique=require_unique,
                    timeout_sec=timeout_sec,
                    node_limit=node_limit,
                    record_trace=False,
                )
                rc_rec = int(
                    rc_res.solved and rc_res.solution is not None and np.array_equal(rc_res.solution, sol)
                )
                raw_rows.append(
                    {
                        "seed": seed,
                        "k": k,
                        "method": "Random-correction",
                        "puzzle_idx": p_idx,
                        "recovery_rate": rc_rec,
                        "timeouts": int(rc_res.limit_hit == "timeout"),
                        "nodes_expanded": rc_res.nodes_expanded,
                        "time_ms": rc_res.time_sec * 1000.0,
                    }
                )
        print(f"  [Exp 3] finished seed={seed}")

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(results_dir / "exp3_raw.csv", index=False)

    seed_df = raw_df.groupby(["k", "method", "seed"], as_index=False, sort=False).agg(
        recovery_rate=("recovery_rate", "mean"),
        nodes_expanded=("nodes_expanded", "mean"),
        timeouts=("timeouts", "sum"),
    )
    summary_df, md_table = _format_mean_std(
        seed_df,
        group_cols=["k", "method"],
        metric_cols=["recovery_rate", "nodes_expanded", "timeouts"],
    )
    summary_df.to_csv(results_dir / "exp3_summary.csv", index=False)
    (results_dir / "exp3_summary.md").write_text(
        f"# Experiment 3: NN-guided vs No Correction vs Random Correction "
        f"(k=0..4, require_unique={require_unique}, {len(seeds)} seeds, {puzzles_per_level} puzzles/level)\n\n"
        + md_table
        + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    styles = [
        ("NN-guided", "#1b9e77", "o"),
        ("Random-correction", "#d95f02", "s"),
        ("No-correction", "#7570b3", "^"),
    ]
    for method, color, marker in styles:
        sub = summary_df[summary_df["method"] == method]
        ax.errorbar(
            sub["k"].to_numpy(),
            sub["recovery_rate_mean"] * 100.0,
            yerr=sub["recovery_rate_std"] * 100.0,
            label=method,
            color=color,
            marker=marker,
            capsize=3,
            linewidth=2.0,
        )
    ax.set_title(f"Exact Solution Recovery vs Misread Level (require_unique={require_unique})")
    ax.set_xlabel("Injected misreads (k)")
    ax.set_ylabel("Exact recovery rate (%)")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "exp3_ablation.png", dpi=150)
    plt.close(fig)

    elapsed = time.perf_counter() - t0
    return summary_df, md_table, elapsed


def write_notes(
    results_dir: Path,
    seeds: list[int],
    puzzles_exp1_3: int,
    puzzles_exp2: int,
    num_givens: int,
    timeout_sec: float,
    node_limit: int,
    exp3_unique: bool,
    timings: dict[str, float],
) -> None:
    """Record exact experimental settings and any scope choices in results/NOTES.md."""
    notes = f"""# Experiment Notes (`eval/run_experiments.py`)

- **Seeds**: `{seeds}` ({len(seeds)} seeds)
- **Exp 1 (`UCS` vs `Greedy` vs `A*`)**:
  - Puzzles per misread level `k in {{0, 1, 2, 3, 4}}`: `{puzzles_exp1_3}` per seed (`{len(seeds) * puzzles_exp1_3}` puzzles per `k`)
  - Givens: `{num_givens}`
  - `max_corrections`: `3` for `k <= 3`, `4` for `k = 4` only
  - Run for both `require_unique=False` and `require_unique=True`
  - Per-puzzle timeout: `{timeout_sec}s`, search node limit: `{node_limit}`
  - Runtime: `{timings.get('exp1', 0.0):.2f}s`
- **Exp 2 (`BT` vs `BT+FC` vs `BT+FC+AC-3`)**:
  - Puzzles per givens level (`25, 30, 35`): `{puzzles_exp2}` per seed (`{len(seeds) * puzzles_exp2}` puzzles per givens level)
  - Per-puzzle timeout: `{timeout_sec}s`
  - Runtime: `{timings.get('exp2', 0.0):.2f}s`
- **Exp 3 (`NN-guided` vs `No-correction` vs `Random-correction`)**:
  - Puzzles per misread level `k in {{0, 1, 2, 3, 4}}`: `{puzzles_exp1_3}` per seed (`{len(seeds) * puzzles_exp1_3}` puzzles per `k`)
  - Givens: `{num_givens}`
  - `require_unique`: `{exp3_unique}`
  - `max_corrections`: `3` for `k <= 3`, `4` for `k = 4` only
  - Per-puzzle timeout: `{timeout_sec}s`, search node limit: `{node_limit}`
  - Runtime: `{timings.get('exp3', 0.0):.2f}s`
"""
    (results_dir / "NOTES.md").write_text(notes, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P7 Experiments 1, 2, and 3.")
    parser.add_argument("--exp", type=str, default="all", choices=("1", "2", "3", "all"))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--puzzles", type=int, default=20, help="Puzzles per level for Exp 1 and 3.")
    parser.add_argument("--csp-puzzles", type=int, default=20, help="Puzzles per givens level for Exp 2.")
    parser.add_argument("--givens", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=2.5, help="Per-puzzle timeout in seconds.")
    parser.add_argument("--node-limit", type=int, default=5000)
    parser.add_argument("--exp3-no-unique", action="store_true", help="Disable require_unique in Exp 3.")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    ctx = _prepare_dataset_and_pools() if args.exp in ("1", "3", "all") else None
    timings: dict[str, float] = {}

    if args.exp in ("1", "all"):
        print("Running Experiment 1...")
        _, md1, t1 = run_exp1(
            seeds=args.seeds,
            puzzles_per_level=args.puzzles,
            num_givens=args.givens,
            timeout_sec=args.timeout,
            node_limit=args.node_limit,
            results_dir=res_dir,
            shared_ctx=ctx,
        )
        timings["exp1"] = t1
        print(f"Exp 1 finished in {t1:.2f}s:\n{md1}\n")

    if args.exp in ("2", "all"):
        print("Running Experiment 2...")
        _, md2, t2 = run_exp2(
            seeds=args.seeds,
            puzzles_per_level=args.csp_puzzles,
            timeout_sec=args.timeout,
            results_dir=res_dir,
        )
        timings["exp2"] = t2
        print(f"Exp 2 finished in {t2:.2f}s:\n{md2}\n")

    if args.exp in ("3", "all"):
        print("Running Experiment 3...")
        _, md3, t3 = run_exp3(
            seeds=args.seeds,
            puzzles_per_level=args.puzzles,
            num_givens=args.givens,
            timeout_sec=args.timeout,
            node_limit=args.node_limit,
            results_dir=res_dir,
            require_unique=not args.exp3_no_unique,
            shared_ctx=ctx,
        )
        timings["exp3"] = t3
        print(f"Exp 3 finished in {t3:.2f}s:\n{md3}\n")

    write_notes(
        results_dir=res_dir,
        seeds=args.seeds,
        puzzles_exp1_3=args.puzzles,
        puzzles_exp2=args.csp_puzzles,
        num_givens=args.givens,
        timeout_sec=args.timeout,
        node_limit=args.node_limit,
        exp3_unique=not args.exp3_no_unique,
        timings=timings,
    )


if __name__ == "__main__":
    main()

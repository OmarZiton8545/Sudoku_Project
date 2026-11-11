# 5-Minute Live Demo Script (`DEMO_SCRIPT.md`)

Run all commands from the project root (`D:\University\Intelligent Systems\Project`) using `.venv\Scripts\python.exe`.

---

### Part 1 — Clean Puzzle Baseline (0:00–1:00)
Show that when there are `0` misreads, AC-3 + CSP solves immediately with `0` search nodes expanded:
```powershell
.venv\Scripts\python.exe run_demo.py --seed 42 --misreads 0
```
- **Point out**: `duplicate_conflicts=[]`, `nodes_expanded=0`, `Correction Log: (none)`, `Equals true solution: True`.

---

### Part 2 — Search Trace & CSP Pruning Comparison (1:00–3:30)
Use **`--seed 43 --misreads 2`**, where **`BT` clearly loses to `BT+FC`** (`503` backtracks vs `0` backtracks) and `A*` solves in `8` traced steps:
```powershell
.venv\Scripts\python.exe run_demo.py --seed 43 --misreads 2 --algo astar --trace
```
- **Point out step by step**:
  1. **AC-3 Conflict Detection**: `status=inconsistent`, detects duplicate `8` at `((0, 3), (4, 3))` and duplicate `4` at `((3, 5), (3, 8))`.
  2. **Search Trace (`--trace`)**: Steps `1..8` show `g`, `h`, and `f = g + h` monotonically guiding `A*` to resolve both conflicts (`h` drops from `0.8887 -> 0.6499 -> 0.0`).
  3. **Correction Log**:
     - `(0, 3): 8 -> 3 (cost=0.6598)`
     - `(3, 8): 4 -> 5 (cost=0.6499)`
     - Final grid matches true solution (`Equals true solution: True`).
  4. **CSP Pruning (`BT` vs `BT+FC` vs `BT+FC+AC-3`)**:
     - `BT`: `554` assignments, **`503` backtracks** (`3.39 ms`)
     - `BT+FC`: `51` assignments, **`0` backtracks** (`0.32 ms`)
     - `BT+FC+AC-3`: `51` assignments, **`0` backtracks** (`1.65 ms`)

---

### Part 3 — NN-On vs NN-Off (`--no-nn`) (3:30–4:30)
Run the exact same broken mosaic with `--no-nn` to skip NN-guided correction search:
```powershell
.venv\Scripts\python.exe run_demo.py --seed 43 --misreads 2 --no-nn
```
- **Point out**:
  - Without NN-guided search, AC-3 immediately fails (`unreadable (--no-nn skipped corrections): no solution: inconsistent puzzle`), and prints the least confident cells (`(0, 3)` confidence `0.3311`, `(3, 8)` confidence `0.5712`) — the exact two cells that `A*` fixed in Part 2.

---

### Part 4 — Visual Walkthrough of Saved Plots (4:30–5:00)
Re-run the traced 2-misread demo (if needed) and open the 4 generated plots in `results/demo/`:
1. `results/demo/input_mosaic.png` — raw `252x252` handwritten digit grid
2. `results/demo/corrected_grid.png` — NN confidence heatmap with red corrected cells
3. `results/demo/csp_backtracks.png` — bar chart showing `BT` losing badly to `BT+FC` and `BT+FC+AC-3`
4. `results/demo/search_nodes.png` — bar chart comparing `UCS`, `Greedy`, and `A*` nodes expanded


# Handwritten Sudoku Reader and Solver (Project P7) — Technical Report

## 1. Problem Statement & Motivation
Optical Sudoku readers often misclassify one or more ambiguous handwritten digits. Even a single misread digit can turn a valid puzzle into an inconsistent or unsolvable constraint satisfaction problem (CSP), causing a standard Sudoku solver to fail.

Project P7 combines three AI techniques into one pipeline:
1. **Calibrated Neural Network (`DigitMLP`)**: Classifies non-blank `28x28` handwritten digit patches (`1–9`) and outputs calibrated probabilities via temperature scaling.
2. **Constraint Satisfaction Problem (`CSPSolver`)**: Checks consistency of the `9x9` grid and solves Sudoku using Backtracking (MRV ordering), Forward Checking (`use_fc`), and Arc Consistency (`use_ac3`).
3. **Correction Search (`UCS`, `Greedy`, `A*`)**: When the initial reading is unsolvable or ambiguous, searches over minimum-penalty cell corrections guided by the NN's probability distribution until the CSP verifies a solvable (or uniquely solvable) puzzle.

---

## 2. System Architecture & Design Choices

### 2.1 Dataset & Preprocessing (`make_data.py`, `data/`)
- **MNIST Digits `1–9`**: Starting from standard MNIST (`60,000` train / `10,000` test, `28x28` grayscale), pixel intensities are scaled to `[0.0, 1.0]`. Digit `0` is dropped first (`5,923` dropped from train, `980` from test) because Sudoku uses digits `1–9`.
- **Single Label-Mapping Helper**: [`digit_to_index`](data/labels.py) and [`index_to_digit`](data/labels.py) map digits `1–9` to class indices `0–8` across all modules.
- **Train / Validation / Test Split**: With `--seed 42`, the `54,077` non-zero training images are split into **`50,000` train** and **`4,077` validation** images; the **`9,020` test** images remain untouched (`data/DATA_DICTIONARY.md`).
- **Blank Cell Rule**: Blank cells are detected by [`is_blank_cell`](data/labels.py) using mean pixel intensity `< 0.01`, not by the neural network. Across all `9,020` non-zero MNIST test digits, the minimum mean pixel intensity is `0.0281`, yielding `0` blank-detection errors across `500` evaluated mosaics (`results/mosaic_injection_report.json`).

### 2.2 Puzzle & Mosaic Generation (`data/sudoku_generator.py`, `integration/reader.py`)
- **Puzzles**: Complete `9x9` grids are built by randomized backtracking (`generate_full_grid`). Cells are removed in random order while preserving uniqueness (`count_solutions(puzzle, limit=2) == 1`) until the target givens (`25`, `30`, or `35`) remain.
- **Post-Training Hard-Digit Injection**: `252x252` mosaics are generated **after** `DigitMLP` is trained. Normal givens sample from `easy_correct` (MNIST test images predicted correctly with confidence `>= 0.90`). To inject `k` misreads (`0..4`), `k` given cells are replaced with MNIST test images of the same true digit from `hard_misread` (test images misclassified by the trained NN, without replacement within a puzzle; fallback to `low_confidence` sorted by highest second-best probability if exhausted).
- Across `100` mosaics per level `k = 0..4` (`30` givens, base seed `42` in `results/mosaic_injection_report.json`), the first NN reading differs from the true puzzle in **exactly `k` cells in `100/100` (`100.0%`)** mosaics at every level, with `0` blank-detection errors.

---

## 3. Neural Network Classifier & Calibration (`nn/`)

### 3.1 Architecture & Training
[`DigitMLP`](nn/model.py) is a 2-hidden-layer MLP (`784 -> 128 -> 64 -> 9` with ReLU activations) trained for `10` epochs (`batch_size=128`, `Adam lr=1e-3`, `--seed 42`) on the `50,000` training images. Temperature scaling (`TemperatureScaler`) fits a single scalar `T > 0` on the `4,077` validation logits via deterministic grid search minimizing validation NLL (`T = 1.2650`).

### 3.2 Test Results vs. Logistic Regression Baseline (`results/nn_metrics.json`)
| Model | Test Accuracy | Test NLL | Test ECE |
| --- | ---: | ---: | ---: |
| `DigitMLP` (before temp scaling) | `0.9772` | `0.0767` | `0.0071` |
| `DigitMLP` (after temp scaling, `T=1.2650`) | `0.9772` | `0.0735` | `0.0023` |
| `LogisticRegression` (`max_iter=500`) | `0.9237` | `0.2774` | `0.0064` |

- Temperature scaling keeps accuracy unchanged (`0.9772`) while reducing test Expected Calibration Error (ECE) by more than **3×** (`0.0071 -> 0.0023`) and improving NLL (`0.0767 -> 0.0735`).
- `DigitMLP` leaves `206 / 9,020` misclassified test digits (13–32 per digit `1–9`), providing real misreads for mosaic injection. Common confusions in `results/nn_confusion_matrix.png` include `4 -> 9` (12), `5 -> 3` (10), `1 -> 8` (9), `2 -> 8` (8), `7 -> 9` (8), `8 -> 4` (8), and `9 -> 4` (8).
- **Plots**: `results/nn_confusion_matrix.png` (9x9 test confusion matrix) and `results/nn_calibration.png` (reliability diagram before/after temperature scaling vs. Logistic Regression).

---

## 4. Constraint Satisfaction Solver (`csp/`)

[`CSPSolver`](csp/sudoku_csp.py) models Sudoku with 81 variables (`0..80`), domains `{1..9}` (fixed singletons for givens), and binary `!=` constraints across each cell's 20 row/column/box peers. It exposes two independent boolean switches:
- **Plain Backtracking (`use_fc=False, use_ac3=False`)**: Blank domains stay `{1..9}`; MRV breaks ties in row-major order and checks peer consistency on assignment.
- **Forward Checking (`use_fc=True, use_ac3=False`)**: Removes every given's digit from unassigned peers' domains before search, then prunes assigned values from unassigned peers during backtracking (restoring on backtrack).
- **AC-3 Preprocessing (`use_ac3=True`)**: Enforces arc consistency across all `1,620` directed peer arcs before search starts, shrinking domains and immediately returning `status="inconsistent"` (`"no solution"`) if two equal givens in a unit empty a domain.
- **Uniqueness Check**: `count_solutions(grid, limit=2)` stops as soon as `2` solutions are found.

---

## 5. Correction Search (`search/`)

### 5.1 State Space & Cost Function
- **State**: `frozenset` of `(cell_index, new_digit)` edits applied to `first_grid` (canonical order-independent representation with `closed` set and `best_g` tracking).
- **Candidate Cell Rule**: Includes **all** non-blank cells currently in a duplicate-digit conflict plus the `10` lowest-confidence non-blank cells (`DEFAULT_LOW_CONF_THRESHOLD = 1.0`). Each candidate cell can change to one of its top-`3` alternative digits ranked by calibrated NN probability.
- **Action Cost**:
  $$\text{cost}(c, d_{\text{new}}) = \max\bigl(0,\; \log P(d_{\text{read}}) - \log P(d_{\text{new}})\bigr)$$
  Path cost is the sum of action costs (`>= 0`). Default budget is `max_corrections = 3` (`4` allowed only for the `k = 4` misread level).
- **Goal Test**: `CSPSolver(use_fc=True, use_ac3=True)` solves the corrected grid (optional `require_unique=True` requires `num_solutions == 1`). If no solvable grid is found within the budget, node limit, or timeout, search returns `"unreadable"` along with the least-confident cells.

### 5.2 Algorithms & Admissibility Proof
- **UCS**: Expands lowest cumulative `g(state)`. Guaranteed optimal because action costs are non-negative.
- **Greedy Best-First**: Priority `(h(state), last_action_cost)`, minimizing remaining conflict heuristic with greedy tie-breaking on the last step cost (ignoring cumulative `g`).
- **A\***: Priority `g(state) + h(state)`. [`conflict_heuristic`](search/correction_search.py) computes `h(state)` over a pairwise cell-disjoint set $P$ of active duplicate-digit conflict pairs $(u, v)$:
  $$h(\text{state}) = \sum_{(u, v) \in P} \min\bigl(\text{min\_alt\_cost}(u),\; \text{min\_alt\_cost}(v)\bigr)$$
  - **Conflict-free grids**: $P = \emptyset \implies h = 0$.
  - **Admissibility ($h \le h^*$)**: Every pair $(u, v) \in P$ is an active duplicate conflict that must be broken to reach solvability. Because pairs in $P$ share no cells, no single cell change can resolve two pairs in $P$, and each uncorrected cell costs at least its smallest top-3 alternative penalty (`>= 0`). Thus $h$ never overestimates remaining cost.

### 5.3 Search Diagnostic (`results/search_diagnostic.json`)
On `20` puzzles/level (`30` givens, `seed=42`, `k=1,2,3`), `UCS` and `A*` had **`0` path-cost mismatches** across all `120` runs (`require_unique=False` and `True`). Failure breakdown (`a`: no conflict detected so `0` corrections accepted; `b`: different cheaper solvable fix accepted; `c`: budget/unreadable):
- `k=1`: `unique=False` recovered `17/20 (85.0%)` (`3` type-`a` failures), `unique=True` recovered `19/20 (95.0%)` (`1` type-`a` failure).
- `k=2`: `unique=False` recovered `13/20 (65.0%)` (`7` type-`b` failures), `unique=True` recovered `14/20 (70.0%)` (`5` type-`b`, `1` type-`c`).
- `k=3`: `unique=False` recovered `11/20 (55.0%)` (`9` type-`b` failures), `unique=True` recovered `18/20 (90.0%)` (`2` type-`b` failures).

---

## 6. End-to-End Integration & Demo (`integration/`, `run_demo.py`)

[`solve_sudoku_mosaic`](integration/pipeline.py) ties together `read_grid` -> initial AC-3 check -> `search_corrections` (skipped under `--no-nn`) -> final CSP verification.

On the traced 2-misread demo (`--seed 43 --misreads 2 --algo astar --trace`, `results/demo/demo_summary.json`):
- **Injected cells**: `(0, 3)` and `(3, 8)`.
- **AC-3 Check**: `status="inconsistent"`, detecting duplicate `8` at `((0, 3), (4, 3))` and duplicate `4` at `((3, 5), (3, 8))`.
- **A\* Search**: Expanded `8` nodes (`path_cost = 1.3098`), logging `(0, 3): 8 -> 3 (cost=0.6598)` and `(3, 8): 4 -> 5 (cost=0.6499)`, and recovered the exact true solution (`exact_recovery=True`).
- **CSP on corrected puzzle**: `BT` took `503` backtracks (`3.39 ms`), whereas `BT+FC` (`0.32 ms`) and `BT+FC+AC-3` (`1.65 ms`) both solved with `0` backtracks.
- **Demo Plots**: `results/demo/input_mosaic.png`, `results/demo/corrected_grid.png`, `results/demo/csp_backtracks.png`, `results/demo/search_nodes.png`.

---

## 7. Multi-Seed Experiments (`eval/run_experiments.py`)

All three experiments used **`5` seeds** (`42, 43, 44, 45, 46`), **`20` puzzles per seed per level** (`100` puzzles total per level), per-puzzle timeout **`2.5s`**, and search node limit **`5000`** (`results/NOTES.md`).

### 7.1 Experiment 1: `UCS` vs `Greedy` vs `A*` (`results/exp1_summary.md`, `results/exp1_search.png`)
| require_unique | k | algo | recovery_rate | nodes_expanded | path_cost | timeouts |
| --- | --- | --- | --- | --- | --- | --- |
| False | 0 | UCS | 100.0% +/- 0.0% | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0 total (0.00 +/- 0.00/seed) |
| False | 0 | GREEDY | 100.0% +/- 0.0% | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0 total (0.00 +/- 0.00/seed) |
| False | 0 | ASTAR | 100.0% +/- 0.0% | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0 total (0.00 +/- 0.00/seed) |
| True | 0 | UCS | 100.0% +/- 0.0% | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0 total (0.00 +/- 0.00/seed) |
| True | 0 | GREEDY | 100.0% +/- 0.0% | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0 total (0.00 +/- 0.00/seed) |
| True | 0 | ASTAR | 100.0% +/- 0.0% | 0.00 +/- 0.00 | 0.00 +/- 0.00 | 0 total (0.00 +/- 0.00/seed) |
| False | 1 | UCS | 80.0% +/- 5.0% | 2.31 +/- 0.29 | 1.44 +/- 0.33 | 0 total (0.00 +/- 0.00/seed) |
| False | 1 | GREEDY | 81.0% +/- 5.5% | 2.67 +/- 1.35 | 1.47 +/- 0.32 | 0 total (0.00 +/- 0.00/seed) |
| False | 1 | ASTAR | 80.0% +/- 5.0% | 2.01 +/- 0.16 | 1.44 +/- 0.33 | 0 total (0.00 +/- 0.00/seed) |
| True | 1 | UCS | 93.0% +/- 2.7% | 3.29 +/- 1.03 | 1.71 +/- 0.23 | 0 total (0.00 +/- 0.00/seed) |
| True | 1 | GREEDY | 93.0% +/- 2.7% | 3.83 +/- 2.06 | 1.71 +/- 0.23 | 0 total (0.00 +/- 0.00/seed) |
| True | 1 | ASTAR | 93.0% +/- 2.7% | 2.65 +/- 0.62 | 1.71 +/- 0.23 | 0 total (0.00 +/- 0.00/seed) |
| False | 2 | UCS | 67.0% +/- 2.7% | 21.52 +/- 15.97 | 3.51 +/- 0.57 | 0 total (0.00 +/- 0.00/seed) |
| False | 2 | GREEDY | 63.0% +/- 4.5% | 12.74 +/- 7.99 | 3.73 +/- 0.68 | 0 total (0.00 +/- 0.00/seed) |
| False | 2 | ASTAR | 67.0% +/- 2.7% | 10.82 +/- 9.40 | 3.51 +/- 0.57 | 0 total (0.00 +/- 0.00/seed) |
| True | 2 | UCS | 89.0% +/- 11.4% | 44.87 +/- 53.47 | 3.88 +/- 0.28 | 0 total (0.00 +/- 0.00/seed) |
| True | 2 | GREEDY | 87.0% +/- 9.7% | 41.05 +/- 53.91 | 4.08 +/- 0.41 | 0 total (0.00 +/- 0.00/seed) |
| True | 2 | ASTAR | 89.0% +/- 11.4% | 34.65 +/- 55.55 | 3.88 +/- 0.28 | 0 total (0.00 +/- 0.00/seed) |
| False | 3 | UCS | 53.0% +/- 8.4% | 72.62 +/- 40.01 | 5.49 +/- 1.16 | 0 total (0.00 +/- 0.00/seed) |
| False | 3 | GREEDY | 46.0% +/- 12.4% | 20.94 +/- 10.35 | 6.07 +/- 1.27 | 0 total (0.00 +/- 0.00/seed) |
| False | 3 | ASTAR | 53.0% +/- 8.4% | 20.31 +/- 9.99 | 5.49 +/- 1.16 | 0 total (0.00 +/- 0.00/seed) |
| True | 3 | UCS | 80.0% +/- 9.4% | 94.01 +/- 32.54 | 6.08 +/- 0.98 | 0 total (0.00 +/- 0.00/seed) |
| True | 3 | GREEDY | 76.0% +/- 10.8% | 31.03 +/- 11.25 | 6.57 +/- 1.00 | 0 total (0.00 +/- 0.00/seed) |
| True | 3 | ASTAR | 80.0% +/- 9.4% | 26.91 +/- 9.18 | 6.08 +/- 0.98 | 0 total (0.00 +/- 0.00/seed) |
| False | 4 | UCS | 37.0% +/- 9.7% | 394.49 +/- 327.41 | 6.65 +/- 1.15 | 0 total (0.00 +/- 0.00/seed) |
| False | 4 | GREEDY | 32.0% +/- 9.1% | 281.77 +/- 308.86 | 7.83 +/- 1.47 | 0 total (0.00 +/- 0.00/seed) |
| False | 4 | ASTAR | 37.0% +/- 9.7% | 142.97 +/- 183.66 | 6.65 +/- 1.15 | 0 total (0.00 +/- 0.00/seed) |
| True | 4 | UCS | 66.0% +/- 6.5% | 543.24 +/- 299.48 | 7.12 +/- 0.88 | 0 total (0.00 +/- 0.00/seed) |
| True | 4 | GREEDY | 59.0% +/- 8.9% | 475.64 +/- 316.45 | 8.57 +/- 2.18 | 0 total (0.00 +/- 0.00/seed) |
| True | 4 | ASTAR | 66.0% +/- 6.5% | 293.38 +/- 230.99 | 7.12 +/- 0.88 | 0 total (0.00 +/- 0.00/seed) |

`A*` matches `UCS`'s exact path cost and recovery at every level while expanding far fewer nodes (e.g., at `k=3, require_unique=True`: `26.91` vs `94.01`; at `k=4, require_unique=True`: `293.38` vs `543.24`). `Greedy` expands fewer nodes than `UCS` on harder grids but pays higher path cost and lower recovery (`59.0%` vs `66.0%` at `k=4, require_unique=True`).

### 7.2 Experiment 2: `BT` vs `BT+FC` vs `BT+FC+AC-3` (`results/exp2_summary.md`, `results/exp2_csp.png`)
| givens | solver | assignments | backtracks | time_ms | timeouts |
| --- | --- | --- | --- | --- | --- |
| 25 | BT | 41974.84 +/- 28252.15 | 41919.38 +/- 28253.26 | 242.69 +/- 157.29 | 2 total (0.40 +/- 0.89/seed) |
| 25 | BT+FC | 315.87 +/- 139.80 | 259.87 +/- 139.80 | 1.31 +/- 0.50 | 0 total (0.00 +/- 0.00/seed) |
| 25 | BT+FC+AC-3 | 315.87 +/- 139.80 | 259.87 +/- 139.80 | 2.51 +/- 0.46 | 0 total (0.00 +/- 0.00/seed) |
| 30 | BT | 5009.64 +/- 2618.98 | 4958.64 +/- 2618.98 | 28.88 +/- 15.50 | 0 total (0.00 +/- 0.00/seed) |
| 30 | BT+FC | 77.72 +/- 11.22 | 26.72 +/- 11.22 | 0.40 +/- 0.04 | 0 total (0.00 +/- 0.00/seed) |
| 30 | BT+FC+AC-3 | 77.72 +/- 11.22 | 26.72 +/- 11.22 | 1.75 +/- 0.16 | 0 total (0.00 +/- 0.00/seed) |
| 35 | BT | 835.77 +/- 509.37 | 789.77 +/- 509.37 | 4.86 +/- 2.95 | 0 total (0.00 +/- 0.00/seed) |
| 35 | BT+FC | 49.34 +/- 2.49 | 3.34 +/- 2.49 | 0.28 +/- 0.02 | 0 total (0.00 +/- 0.00/seed) |
| 35 | BT+FC+AC-3 | 49.34 +/- 2.49 | 3.34 +/- 2.49 | 1.65 +/- 0.02 | 0 total (0.00 +/- 0.00/seed) |

### 7.3 Experiment 3: `NN-guided` vs `No-correction` vs `Random-correction` (`results/exp3_summary.md`, `results/exp3_ablation.png`)
| k | method | recovery_rate | nodes_expanded | timeouts |
| --- | --- | --- | --- | --- |
| 0 | NN-guided | 100.0% +/- 0.0% | 0.00 +/- 0.00 | 0 total (0.00 +/- 0.00/seed) |
| 0 | No-correction | 100.0% +/- 0.0% | 0.00 +/- 0.00 | 0 total (0.00 +/- 0.00/seed) |
| 0 | Random-correction | 100.0% +/- 0.0% | 0.00 +/- 0.00 | 0 total (0.00 +/- 0.00/seed) |
| 1 | NN-guided | 93.0% +/- 2.7% | 2.65 +/- 0.62 | 0 total (0.00 +/- 0.00/seed) |
| 1 | No-correction | 0.0% +/- 0.0% | 0.00 +/- 0.00 | 0 total (0.00 +/- 0.00/seed) |
| 1 | Random-correction | 25.0% +/- 7.1% | 684.35 +/- 256.46 | 0 total (0.00 +/- 0.00/seed) |
| 2 | NN-guided | 89.0% +/- 11.4% | 34.65 +/- 55.55 | 0 total (0.00 +/- 0.00/seed) |
| 2 | No-correction | 0.0% +/- 0.0% | 0.00 +/- 0.00 | 0 total (0.00 +/- 0.00/seed) |
| 2 | Random-correction | 6.0% +/- 4.2% | 2478.03 +/- 480.50 | 0 total (0.00 +/- 0.00/seed) |
| 3 | NN-guided | 80.0% +/- 9.4% | 26.91 +/- 9.18 | 0 total (0.00 +/- 0.00/seed) |
| 3 | No-correction | 0.0% +/- 0.0% | 0.00 +/- 0.00 | 0 total (0.00 +/- 0.00/seed) |
| 3 | Random-correction | 0.0% +/- 0.0% | 3926.59 +/- 193.69 | 0 total (0.00 +/- 0.00/seed) |
| 4 | NN-guided | 66.0% +/- 6.5% | 293.38 +/- 230.99 | 0 total (0.00 +/- 0.00/seed) |
| 4 | No-correction | 0.0% +/- 0.0% | 0.00 +/- 0.00 | 0 total (0.00 +/- 0.00/seed) |
| 4 | Random-correction | 0.0% +/- 0.0% | 4356.36 +/- 158.69 | 24 total (4.80 +/- 1.48/seed) |

---

## 8. Limitations & Methodological Notes
1. **Synthetic Mosaics, Not Photographed Sudoku**: Inputs are `252x252` grayscale composites of centered `28x28` MNIST digits and pure-zero blank tiles, not camera-captured grids with lighting, perspective skew, or grid-line artifacts.
2. **Injected Misreads vs. Natural Frequency**: Misread digits (`k=0..4`) are deliberately injected from `hard_misread` (MNIST test images the NN gets wrong). Experiments therefore measure **recovery conditional on `k` misreads**, not natural end-to-end misread frequency on random MNIST boards.
3. **Candidate-Cell Rule Tuning**: `DEFAULT_LOW_CONF_THRESHOLD = 1.0` (always taking the top-`10` lowest-confidence cells plus all conflict cells) was tuned after inspecting `20` puzzles in the initial `k=1` check, where a `0.989`-confidence misread (`i=17`) had been excluded by `0.98`.
4. **Demo Seed Selection**: Demo seeds (`42` for `k=0,1` and `43` for `k=2`) were selected specifically to show clean AC-3 conflict detection and successful recovery.
5. **Initial `BT+FC` Bug & Exp 2 Rerun**: An initial `BT+FC` implementation did not prune givens from peer domains up front, causing MRV to chase peers of early guesses and making `BT+FC` worse than plain `BT` (`52616.65` vs `41756.40` backtracks at 25 givens; `6026.63` vs `4958.64` at 30 givens; archived in `results/old_exp2/`). `csp/fc_diagnostic.py` (`results/fc_diagnostic.json`) isolated the bug, `csp/sudoku_csp.py` was fixed, and Exp 2 was rerun.
6. **`BT+FC` vs `BT+FC+AC-3` Equivalence on Solvable Puzzles**: In the final Exp 2, `BT+FC` and `BT+FC+AC-3` have identical `assignments` and `backtracks` (`259.87` at 25 givens, `26.72` at 30 givens, `3.34` at 35 givens) because `BT+FC`'s upfront given pruning matches AC-3's first pass on Sudoku; AC-3 therefore only adds preprocessing time on valid grids (`2.51 ms` vs `1.31 ms` at 25 givens). AC-3's practical value in this system is **early conflict detection** (`"inconsistent"`, `0` assignments/backtracks) on broken grids.
7. **Timeouts**: With a `2.5s` per-puzzle timeout, `BT` timed out `2` times (`0.40 +/- 0.89`/seed) at `25` givens in Exp 2, and `Random-correction` timed out `24` times (`4.80 +/- 1.48`/seed) at `k=4` in Exp 3.
8. **Experiment Sample Size**: All three experiments used `5` seeds (`42..46`) × `20` puzzles per level (`100` puzzles total per level across seeds), rather than `100` puzzles per seed (`500` total), to keep total runtime (~10.0 min) under 20 minutes (`results/NOTES.md`).
9. **Correction Budget**: Maximum corrections is `3` by default, with `max_corrections = 4` enabled only for the `k = 4` misread level.
10. **Uniqueness vs. Search Cost**: Enabling `require_unique=True` raises exact recovery (e.g., Exp 1 UCS/A\* `k=3`: `53.0% -> 80.0%`; `k=4`: `37.0% -> 66.0%`), at the cost of expanding more nodes (`k=3` UCS: `72.62 -> 94.01`; `k=4` UCS: `394.49 -> 543.24`).

---

## 9. Conclusions
Combining a temperature-scaled digit classifier (`97.72%` test accuracy, `0.0023` ECE), AC-3/FC Sudoku solving (`26.72` vs `4958.64` backtracks at 30 givens), and admissible `A*` correction search yields a reliable, interpretable reader-and-solver pipeline. NN-guided `A*` with `require_unique=True` recovers `93.0%`, `89.0%`, `80.0%`, and `66.0%` of puzzles across `1–4` injected misreads, compared to `0.0%` without correction and `0.0%–25.0%` under random correction.


# 10-Minute Presentation Outline (`SLIDES.md`)

**Target pacing:** ~10 slides (~1 minute per slide), followed by the 5-minute live demo (`DEMO_SCRIPT.md`).

---

### Slide 1 — Title & Motivation (0:00–1:00)
- **Title**: Handwritten Sudoku Reader and Solver (Project P7)
- **Problem**: One misread handwritten digit breaks an entire Sudoku puzzle.
- **Core Idea**: Combine a calibrated Neural Network + Search + CSP so uncertain NN predictions become soft costs rather than hard errors.
- **Figure**: `results/demo/input_mosaic.png`

### Slide 2 — End-to-End Pipeline (1:00–2:00)
- **Flow**: `252x252` mosaic -> brightness blank rule (`mean < 0.01`, not NN) -> `DigitMLP` calibrated probabilities -> AC-3 conflict check -> `A*` / `UCS` correction search -> `CSPSolver` solution.
- **Dataset**: MNIST digits `1–9` (`50,000` train / `4,077` val / `9,020` test; `0` dropped first).
- **Figure**: `results/demo/corrected_grid.png`

### Slide 3 — Neural Network & Calibration (2:00–3:00)
- **Model**: `DigitMLP` (`784 -> 128 -> 64 -> 9`) vs `LogisticRegression`.
- **Accuracy**: `97.72%` vs `92.37%` (`206 / 9,020` test misreads).
- **Temperature Scaling (`T=1.2650`)**: ECE drops `0.0071 -> 0.0023` (over 3× better calibration; NLL `0.0767 -> 0.0735`).
- **Figure**: `results/nn_calibration.png`

### Slide 4 — Error Patterns & Hard-Digit Injection (3:00–4:00)
- **Confusion Matrix**: Real MNIST errors (e.g., `4->9`, `5->3`, `1->8`).
- **Mosaic Injection**: `100/100` exact `k`-misread rates for `k=0..4` and `0` blank errors (`results/mosaic_injection_report.json`).
- **Figure**: `results/nn_confusion_matrix.png`

### Slide 5 — Sudoku CSP & Switchable Pruning (4:00–5:00)
- **Formulation**: `81` cells, domains `1–9`, row/col/box `!=` constraints, MRV ordering.
- **Switches**: Plain `BT` vs `BT+FC` (upfront given pruning + live peer pruning) vs `BT+FC+AC-3` (arc-consistency preprocessing + early conflict detection).
- **Figure**: `results/demo/csp_backtracks.png`

### Slide 6 — Exp 2: CSP Pruning Results (5:00–6:00)
- **25/30/35 givens (5 seeds × 20 puzzles)**:
  - `30` givens: `BT` `4958.64` backtracks -> `BT+FC` `26.72` -> `BT+FC+AC-3` `26.72`.
  - `25` givens: `BT` `41919.38` (`2` timeouts) -> `BT+FC` `259.87` (`1.31 ms`) & `BT+FC+AC-3` `259.87` (`2.51 ms`).
- **Key insight**: On valid Sudoku grids FC's initial pruning matches AC-3's first pass; AC-3's key advantage here is instant conflict detection (`status="inconsistent"`).
- **Figure**: `results/exp2_csp.png`

### Slide 7 — Correction Search: UCS, Greedy & Admissible A\* (6:00–7:00)
- **State**: `frozenset` of `(cell, new_digit)`, max `3` corrections (`4` for `k=4`).
- **Cost**: `max(0, log(p_read) - log(p_new))` (non-negative).
- **A\* Heuristic**: Sum of minimum correction costs over cell-disjoint duplicate conflict pairs (`h = 0` when conflict-free; never overestimates).
- **Figure**: `results/demo/search_nodes.png`

### Slide 8 — Exp 1: Search Comparison & Uniqueness Goal (7:00–8:00)
- **Optimality**: `0` path-cost mismatches between `UCS` and `A*` (`results/search_diagnostic.json`).
- **Efficiency**: At `k=3` (`require_unique=True`), `A*` expands `26.91` nodes vs `UCS` `94.01` while achieving identical `80.0%` recovery.
- **Uniqueness effect**: `require_unique=True` boosts `k=3` recovery `53.0% -> 80.0%` and `k=4` `37.0% -> 66.0%` by rejecting ambiguous cheaper fixes.
- **Figure**: `results/exp1_search.png`

### Slide 9 — Exp 3: NN-Guided vs Baselines (8:00–9:00)
- **Ablation (`k=0..4`, `require_unique=True`)**:
  - `NN-guided`: `100% -> 93% -> 89% -> 80% -> 66%` recovery (`0` timeouts).
  - `No-correction`: `0%` for all `k >= 1`.
  - `Random-correction`: `25%` (`k=1`), `6%` (`k=2`), `0%` (`k=3,4`; `24` timeouts at `k=4`).
- **Figure**: `results/exp3_ablation.png`

### Slide 10 — Limits, Lessons Learned & Conclusion (9:00–10:00)
- **Honest Limits**: Synthetic `252x252` mosaics (not camera photos); injected misreads measure recovery given `k`; candidate-cell rule tuned on 20 puzzles; `BT+FC` initial bug caught via diagnostic (`results/old_exp2/`).
- **Takeaway**: Calibrated NN probabilities + admissible `A*` + AC-3/FC CSP turns brittle optical digit reading into a `66%–93%` recoverable solver.
- **Figure**: `results/exp3_ablation.png` (or `results/demo/corrected_grid.png`)


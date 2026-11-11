# Experiment Notes (`eval/run_experiments.py`)

- **Seeds**: `[42, 43, 44, 45, 46]` (5 seeds)
- **Scope / Speed-Mode Note**:
  - Each experiment used **`20` puzzles per seed per level** across **`5` seeds** (`100` total puzzles per level per configuration).
  - Original full sweep runtime was **`600.43s` (~10.0 min)**: Exp 1 (`153.60s`) + original Exp 2 (`100.34s`) + Exp 3 (`346.49s`).
  - Scaling Exp 1 + Exp 3 to `100` puzzles/seed (`5x` = `500` puzzles per level) would take `~41.7 min` (exceeding the 20-minute budget, primarily due to `Random-correction` hitting `2.5s` timeouts / `5000`-node limits at `k=3, 4`), so we kept the `20` puzzles/seed × `5` seeds (`100` total puzzles/level) dataset and documented this reduction explicitly here.

## Exp 1 (`UCS` vs `Greedy` vs `A*`) — Unaffected (`use_fc=True, use_ac3=True`)
- Puzzles per misread level `k in {0, 1, 2, 3, 4}`: `20` per seed (`100` total per `k`), `30` givens
- `max_corrections`: `3` for `k <= 3`, `4` for `k = 4` only
- Run for both `require_unique=False` and `require_unique=True`
- Per-puzzle timeout: `2.5s`, search node limit: `5000`
- Timeouts observed: `0` across all `k`, algorithms, and uniqueness settings
- Runtime: `153.60s`

## Exp 2 (`BT` vs `BT+FC` vs `BT+FC+AC-3`) — Updated After `BT+FC` Fix
- Puzzles per givens level (`25, 30, 35`): `20` per seed (`100` total per givens level), per-puzzle timeout `2.5s`
- **What was wrong (archived in `results/old_exp2/`)**:
  - When `use_fc=True` and `use_ac3=False`, [`CSPSolver.solve`](../csp/sudoku_csp.py) initialized blank domains to `{1..9}` without pruning the initial givens from peer domains. As a result, `BT+FC`'s MRV heuristic jumped to peers of early guessed cells whose domains still contained digits forbidden by the givens, causing higher variance and more backtracks on average than plain `BT` (`52616.65` vs `41756.40` at 25 givens; `6026.63` vs `4958.64` at 30 givens).
- **What was fixed**:
  - When `use_fc=True` (even with `use_ac3=False`), `CSPSolver.solve` now prunes every given's digit from its row, column, and box peers before search (counting these removals in `values_pruned`). Plain `BT` (`use_fc=False`) and `BT+FC+AC-3` (`use_ac3=True`) remain unchanged, so Exp 1 and Exp 3 are unaffected.
- **Old vs New Exp 2 (`BT+FC` mean backtracks / timeouts)**:
  - `25 givens`: Old `52616.65 +/- 27972.78` (`4` timeouts) -> New `259.87 +/- 139.80` (`0` timeouts); `BT` = `41919.38 +/- 28253.26` (`2` timeouts)
  - `30 givens`: Old `6026.63 +/- 2126.46` (`0` timeouts) -> New `26.72 +/- 11.22` (`0` timeouts); `BT` = `4958.64 +/- 2618.98`
  - `35 givens`: Old `769.39 +/- 522.85` (`0` timeouts) -> New `3.34 +/- 2.49` (`0` timeouts); `BT` = `789.77 +/- 509.37`
- New Exp 2 Runtime: `52.50s`

## Exp 3 (`NN-guided` vs `No-correction` vs `Random-correction`) — Unaffected (`use_fc=True, use_ac3=True`)
- Puzzles per misread level `k in {0, 1, 2, 3, 4}`: `20` per seed (`100` total per `k`), `30` givens
- `require_unique`: `True`
- `max_corrections`: `3` for `k <= 3`, `4` for `k = 4` only
- Per-puzzle timeout: `2.5s`, search node limit: `5000`
- Timeouts observed: `0` on `NN-guided`, `0` on `No-correction`, `24` on `Random-correction` at `k = 4`
- Runtime: `346.49s`

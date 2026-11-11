# Q&A Preparation (`QA_PREP.md`)

1. **Why does Forward Checking (`BT+FC`) reduce backtracks so drastically compared to plain `BT`?**
   Because `BT+FC` prunes every given's digit from its peers' domains before search and removes assigned values from unassigned peers during search. That exposes forced singles (`domain size == 1`) immediately to MRV instead of guessing blindly over `{1..9}` (`4958.64 -> 26.72` backtracks at 30 givens in Exp 2).

2. **Why does `BT+FC+AC-3` have the exact same assignments and backtracks as `BT+FC` in Exp 2?**
   Because on valid Sudoku puzzles, `BT+FC`'s upfront peer pruning from givens produces the same initial domains as AC-3's first pass (`len(domain) == 1` arc revisions), and MRV picks size-1 variables next anyway. Thus `AC-3` only adds preprocessing time on solvable grids (`2.51 ms` vs `1.31 ms` at 25 givens); its value here is **early conflict detection** (`status="inconsistent"` with `0` assignments) on misread grids.

3. **Why did `BT+FC` originally have *more* backtracks than `BT` before your fix?**
   Originally, `use_fc=True` did not prune initial givens from peer domains before search, so MRV jumped to peers of early guessed cells whose domains still held invalid digits forbidden by the givens. We caught this with `csp/fc_diagnostic.py`, fixed `CSPSolver.solve`, reran Exp 2, and archived the old results in `results/old_exp2/`.

4. **Why is the action cost `max(0, log(p_read) - log(p_new))` (difference of negative log-probabilities)?**
   Maximizing the joint probability $\prod_i P(d_i)$ of the board is equivalent to minimizing $\sum_i -\log P(d_i)$. Relative to the baseline first reading, changing cell $c$ from $d_{\text{read}}$ to $d_{\text{new}}$ adds penalty $(-\log P(d_{\text{new}})) - (-\log P(d_{\text{read}})) = \log P(d_{\text{read}}) - \log P(d_{\text{new}}) \ge 0$, making path costs additive and non-negative.

5. **Why is Uniform Cost Search (UCS) guaranteed to find the cheapest correction set?**
   Every action cost is $\ge 0$ (`max(0.0, ...)`), states are canonical `frozenset`s with a `closed` set and `best_g` check, and the goal test runs on pop. Therefore, the first goal state popped by UCS is guaranteed to have minimum total path cost.

6. **Why does the `A*` heuristic `h(state)` never overestimate (`h <= h*`)?**
   `conflict_heuristic` collects a pairwise **cell-disjoint** set $P$ of active duplicate-digit conflict pairs $(u, v)$. Every pair $(u, v) \in P$ must have at least one cell changed to become conflict-free, and because pairs in $P$ share no cells, no single edit can resolve two pairs in $P$. Summing $\min(\text{min\_alt\_cost}(u), \text{min\_alt\_cost}(v))$ over $P$ is therefore a strict lower bound on remaining cost, and returns `0.0` on any conflict-free grid.

7. **Why does `require_unique=True` improve exact recovery (`53% -> 80%` at `k=3`)?**
   With `require_unique=False`, changing fewer cells than `k` can accidentally yield an under-constrained puzzle that still has *some* valid Sudoku solution—just not the true one (`results/search_diagnostic.json` showed `9/9` `k=3` failures were cheaper wrong solvable fixes). Requiring a unique solution rejects those ambiguous grids so search continues to the true correction, at the cost of extra nodes (`94.01` vs `72.62` for UCS at `k=3`).

8. **Why does temperature scaling improve calibration without changing accuracy?**
   Dividing logits by a positive scalar $T$ (`T=1.2650` fitted on the `4,077` validation images) preserves the `argmax` order of classes, so test accuracy stays `0.9772`, while softening overconfident logits lowers test ECE from `0.0071` to `0.0023` (`results/nn_metrics.json`).

9. **Why not use the neural network to detect blank cells?**
   A brightness rule (`mean < 0.01`) is simpler, deterministic, and avoids forcing a `1–9` classifier to guess on empty tiles. Across all `9,020` non-zero MNIST test digits, the minimum mean brightness is `0.0281`, so blank detection had `0` errors across all `500` mosaics (`results/mosaic_injection_report.json`).

10. **Why did you drop digit `0` before splitting train and validation?**
    Sudoku uses digits `1–9` only (`9` classes mapped via `digit_to_index` to `0–8`). Dropping `0` first (`5,923` from train, `980` from test) keeps the `50,000 / 4,077 / 9,020` split purely over valid Sudoku digits while leaving the test set untouched.

11. **Why did `Greedy` occasionally expand slightly *more* nodes than `A*` (e.g., `31.03` vs `26.91` at `k=3, unique=True`)?**
    `Greedy` uses priority `(h, last_action_cost)` and ignores cumulative `g`. When multiple states have `h = 0` or equal `h`, Greedy can explore a locally tempting dead-end branch that `A*` avoids because `g + h` penalizes high cumulative cost.

12. **Why does recovery drop from `93%` (`k=1`) to `66%` (`k=4`) even with `require_unique=True`?**
    Two reasons: (1) an injected misread's true digit may rank outside the top-3 alternatives or the cell may fall outside the top-10 lowest-confidence candidates, and (2) with `4` wrong givens, another uniquely solvable puzzle can occasionally exist at a lower log-probability penalty.

13. **Why is `max_corrections=4` allowed only for `k=4`?**
    By default the solver caps edits at `3` (`DEFAULT_MAX_CORRECTIONS = 3`) to bound the combinatorial branching factor ($\binom{n}{k} \cdot 3^k$) and prevent excessive hallucinated edits. Level `k=4` in Exp 1/3 specifically raises the budget to `4` so recovering `4` injected errors is possible.

14. **Why did `Random-correction` time out `24` times at `k=4` in Exp 3?**
    Without NN confidence or digit ranking, `Random-correction` expands blind subsets across arbitrary non-blank cells and random digits, hitting the `5000`-node / `2.5s` per-puzzle budget at `k=4` (`0.0%` recovery) while `NN-guided` `A*` averages `293.38` nodes with `0` timeouts (`66.0%` recovery).

15. **Could this system run directly on a phone photo of a newspaper Sudoku?**
    Not without a computer-vision front end: our inputs are clean `252x252` MNIST composites with zero-intensity blank tiles and no perspective distortion, grid lines, or lighting noise, and our misreads are injected from `hard_misread` rather than measured on photographed digits.


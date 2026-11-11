"""Seeded Sudoku puzzle generation and NN-guided MNIST mosaic builder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.labels import DIGITS, digit_to_index, index_to_digit


@dataclass
class DigitPools:
    """MNIST test indices grouped by true digit (1..9) into easy and hard pools."""

    easy_correct: dict[int, np.ndarray]
    hard_misread: dict[int, np.ndarray]
    low_confidence: dict[int, np.ndarray]


def _candidates(grid: np.ndarray, r: int, c: int) -> list[int]:
    """Return valid digits 1..9 for empty cell (r, c)."""
    used = set(grid[r, :]) | set(grid[:, c])
    br, bc = 3 * (r // 3), 3 * (c // 3)
    used |= set(grid[br : br + 3, bc : bc + 3].ravel())
    return [d for d in DIGITS if d not in used]


def generate_full_grid(rng: np.random.Generator) -> np.ndarray:
    """Create a complete valid 9x9 Sudoku grid by randomized backtracking."""
    grid = np.zeros((9, 9), dtype=np.int64)

    def _fill(pos: int) -> bool:
        if pos == 81:
            return True
        r, c = divmod(pos, 9)
        digits = list(DIGITS)
        rng.shuffle(digits)
        used_row = set(grid[r, :c])
        used_col = set(grid[:r, c])
        br, bc = 3 * (r // 3), 3 * (c // 3)
        used_box = set(grid[br : br + 3, bc : bc + 3].ravel())
        used = used_row | used_col | used_box
        for d in digits:
            if d not in used:
                grid[r, c] = d
                if _fill(pos + 1):
                    return True
                grid[r, c] = 0
        return False

    _fill(0)
    return grid


def count_solutions_fast(grid: np.ndarray, limit: int = 2) -> int:
    """Count Sudoku solutions up to `limit` using MRV backtracking."""
    work = grid.copy()
    row_mask = [0] * 9
    col_mask = [0] * 9
    box_mask = [0] * 9
    empties: list[tuple[int, int]] = []

    for r in range(9):
        for c in range(9):
            val = int(work[r, c])
            if val == 0:
                empties.append((r, c))
            else:
                bit = 1 << val
                b = 3 * (r // 3) + (c // 3)
                if (row_mask[r] & bit) or (col_mask[c] & bit) or (box_mask[b] & bit):
                    return 0
                row_mask[r] |= bit
                col_mask[c] |= bit
                box_mask[b] |= bit

    all_bits = 0b1111111110  # bits 1..9

    def _solve(remaining: list[tuple[int, int]]) -> int:
        if not remaining:
            return 1
        # Pick empty cell with fewest valid candidates (MRV).
        best_idx = -1
        best_avail = 0
        best_count = 10
        for idx, (r, c) in enumerate(remaining):
            b = 3 * (r // 3) + (c // 3)
            avail = all_bits & ~(row_mask[r] | col_mask[c] | box_mask[b])
            cnt = avail.bit_count()
            if cnt == 0:
                return 0
            if cnt < best_count:
                best_count = cnt
                best_avail = avail
                best_idx = idx
                if cnt == 1:
                    break

        r, c = remaining[best_idx]
        b = 3 * (r // 3) + (c // 3)
        next_rem = remaining[:best_idx] + remaining[best_idx + 1 :]
        total = 0
        avail = best_avail
        while avail:
            lsb = avail & -avail
            avail ^= lsb
            row_mask[r] |= lsb
            col_mask[c] |= lsb
            box_mask[b] |= lsb
            total += _solve(next_rem)
            row_mask[r] ^= lsb
            col_mask[c] ^= lsb
            box_mask[b] ^= lsb
            if total >= limit:
                return total
        return total

    return _solve(empties)


def generate_puzzle(
    num_givens: int = 30,
    rng: np.random.Generator | None = None,
    seed: int = 42,
    max_attempts: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a (puzzle, solution) pair with `num_givens` clues and unique solution."""
    if not 17 <= num_givens <= 81:
        raise ValueError(f"num_givens must be in 17..81, got {num_givens}")
    if rng is None:
        rng = np.random.default_rng(seed)

    for _ in range(max_attempts):
        solution = generate_full_grid(rng)
        puzzle = solution.copy()
        cells = [(r, c) for r in range(9) for c in range(9)]
        rng.shuffle(cells)

        givens = 81
        for r, c in cells:
            if givens <= num_givens:
                break
            backup = puzzle[r, c]
            puzzle[r, c] = 0
            if count_solutions_fast(puzzle, limit=2) != 1:
                puzzle[r, c] = backup
            else:
                givens -= 1

        if givens == num_givens:
            return puzzle, solution

    raise RuntimeError(f"Failed to generate puzzle with exactly {num_givens} givens")


def build_hard_digit_pools(
    test_labels: np.ndarray,
    probs: np.ndarray,
    easy_conf_threshold: float = 0.90,
) -> DigitPools:
    """Group MNIST test indices per digit using trained NN probabilities.

    - easy_correct: correctly predicted digits with confidence >= easy_conf_threshold.
    - hard_misread: test images actually misread by the NN, sorted by highest
      probability assigned to the true digit (realistic borderline misreads).
    - low_confidence: lowest-confidence images for fallback if misread pool is small.
    """
    pred_idx = np.argmax(probs, axis=1)
    top_conf = np.max(probs, axis=1)

    easy_correct: dict[int, np.ndarray] = {}
    hard_misread: dict[int, np.ndarray] = {}
    low_confidence: dict[int, np.ndarray] = {}

    for d in DIGITS:
        c_idx = digit_to_index(d)
        mask_d = test_labels == c_idx
        indices_d = np.where(mask_d)[0]

        correct_mask = pred_idx[indices_d] == c_idx
        easy_mask = correct_mask & (top_conf[indices_d] >= easy_conf_threshold)
        easy = indices_d[easy_mask]
        if len(easy) == 0:
            # Fallback to highest-confidence correct samples for this digit.
            correct_only = indices_d[correct_mask]
            order = np.argsort(-top_conf[correct_only])
            easy = correct_only[order[: max(1, len(correct_only) // 2)]]
        easy_correct[d] = easy

        misread = indices_d[~correct_mask]
        # Prefer misreads where the true digit still has relatively high NN probability.
        true_probs = probs[misread, c_idx]
        misread_sorted = misread[np.argsort(-true_probs)]
        hard_misread[d] = misread_sorted

        # Fallback low-confidence pool sorted by smallest margin / top confidence.
        low_sorted = indices_d[np.argsort(top_conf[indices_d])]
        low_confidence[d] = low_sorted[: max(10, len(low_sorted) // 10)]

    return DigitPools(
        easy_correct=easy_correct,
        hard_misread=hard_misread,
        low_confidence=low_confidence,
    )


def build_mosaic(
    puzzle: np.ndarray,
    test_images: np.ndarray,
    pools: DigitPools,
    num_misreads: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build a (9, 9, 28, 28) mosaic from MNIST test digits with `num_misreads` hard digits."""
    given_coords = [(int(r), int(c)) for r, c in np.argwhere(puzzle > 0)]
    if num_misreads > len(given_coords):
        raise ValueError(
            f"num_misreads={num_misreads} exceeds number of givens ({len(given_coords)})"
        )

    chosen_idx = rng.choice(len(given_coords), size=num_misreads, replace=False)
    misread_set = {given_coords[i] for i in chosen_idx}

    mosaic = np.zeros((9, 9, 28, 28), dtype=np.float32)
    for r, c in given_coords:
        digit = int(puzzle[r, c])
        if (r, c) in misread_set:
            hard_pool = pools.hard_misread[digit]
            if len(hard_pool) > 0:
                # Sample from the top half of plausible misreads so true digit stays in top-k.
                cap = max(1, min(len(hard_pool), 40))
                img_idx = int(hard_pool[int(rng.integers(0, cap))])
            else:
                low_pool = pools.low_confidence[digit]
                img_idx = int(low_pool[int(rng.integers(0, len(low_pool)))])
        else:
            easy_pool = pools.easy_correct[digit]
            img_idx = int(easy_pool[int(rng.integers(0, len(easy_pool)))])
        mosaic[r, c] = test_images[img_idx]

    misread_cells = sorted(misread_set)
    return mosaic, misread_cells


def mosaic_to_2d_image(mosaic: np.ndarray) -> np.ndarray:
    """Tile a (9, 9, 28, 28) mosaic into a single (252, 252) grayscale image."""
    return mosaic.transpose(0, 2, 1, 3).reshape(252, 252)


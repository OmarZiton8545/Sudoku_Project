"""Shared helpers for digit-to-class mapping and blank cell detection."""

from __future__ import annotations

import numpy as np

# Valid Sudoku digits 1..9 correspond to class indices 0..8.
DIGITS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
NUM_CLASSES = 9
DEFAULT_BLANK_THRESHOLD = 0.01


def digit_to_index(digit: int) -> int:
    """Map a Sudoku digit (1-9) to a 0-based class index (0-8)."""
    if not 1 <= int(digit) <= 9:
        raise ValueError(f"Expected digit in 1..9, got {digit}")
    return int(digit) - 1


def index_to_digit(index: int) -> int:
    """Map a 0-based class index (0-8) back to a Sudoku digit (1-9)."""
    if not 0 <= int(index) <= 8:
        raise ValueError(f"Expected class index in 0..8, got {index}")
    return int(index) + 1


def is_blank_cell(patch: np.ndarray, threshold: float = DEFAULT_BLANK_THRESHOLD) -> bool:
    """Return True if a 28x28 cell image is blank based on mean pixel brightness."""
    return float(np.mean(patch)) < threshold


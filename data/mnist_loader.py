"""MNIST download, digit-0 removal, and seeded train/val/test preparation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torchvision import datasets

from data.labels import NUM_CLASSES, digit_to_index


TRAIN_SPLIT_SIZE = 50_000


def _filter_and_scale(images: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Scale uint8 images to [0, 1], drop digit 0, and map digits 1-9 to 0-8."""
    keep_mask = targets != 0
    dropped_zeros = int(np.sum(~keep_mask))

    x = images[keep_mask].astype(np.float32) / 255.0
    raw_digits = targets[keep_mask]
    # Use the shared helper to map digits 1..9 -> class indices 0..8.
    y = np.array([digit_to_index(int(d)) for d in raw_digits], dtype=np.int64)
    return x, y, dropped_zeros


def prepare_mnist(
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    seed: int = 42,
    train_size: int = TRAIN_SPLIT_SIZE,
) -> dict:
    """Download MNIST, drop digit 0, split train/val, and save processed arrays."""
    raw_path = Path(raw_dir)
    out_path = Path(processed_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.MNIST(root=str(raw_path), train=True, download=True)
    test_ds = datasets.MNIST(root=str(raw_path), train=False, download=True)

    x_full, y_full, train_zeros = _filter_and_scale(
        train_ds.data.numpy(), train_ds.targets.numpy()
    )
    x_test, y_test, test_zeros = _filter_and_scale(
        test_ds.data.numpy(), test_ds.targets.numpy()
    )

    if train_size >= len(x_full):
        raise ValueError(
            f"train_size={train_size} must be smaller than non-zero train count {len(x_full)}"
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_full))
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    x_train, y_train = x_full[train_idx], y_full[train_idx]
    x_val, y_val = x_full[val_idx], y_full[val_idx]

    npz_file = out_path / "mnist_split.npz"
    np.savez_compressed(
        npz_file,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    meta = {
        "seed": seed,
        "raw_train_total": int(len(train_ds)),
        "raw_test_total": int(len(test_ds)),
        "train_zeros_dropped": train_zeros,
        "test_zeros_dropped": test_zeros,
        "train_size": int(len(x_train)),
        "val_size": int(len(x_val)),
        "test_size": int(len(x_test)),
        "num_classes": NUM_CLASSES,
        "image_shape": [28, 28],
        "train_class_counts": [int(np.sum(y_train == c)) for c in range(NUM_CLASSES)],
        "val_class_counts": [int(np.sum(y_val == c)) for c in range(NUM_CLASSES)],
        "test_class_counts": [int(np.sum(y_test == c)) for c in range(NUM_CLASSES)],
    }
    meta_file = out_path / "mnist_meta.json"
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_processed_mnist(processed_dir: str | Path = "data/processed") -> dict[str, np.ndarray]:
    """Load preprocessed train/val/test arrays from disk."""
    npz_file = Path(processed_dir) / "mnist_split.npz"
    if not npz_file.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {npz_file}. Run make_data.py first."
        )
    with np.load(npz_file) as data:
        return {key: data[key] for key in data.files}


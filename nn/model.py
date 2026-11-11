"""PyTorch MLP digit classifier (digits 1-9 -> classes 0-8) and temperature scaling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from data.labels import NUM_CLASSES


class DigitMLP(nn.Module):
    """Simple 2-hidden-layer MLP for 28x28 handwritten digits 1-9."""

    def __init__(self, hidden1: int = 128, hidden2: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemperatureScaler:
    """Simple 1-parameter post-hoc calibration: softmax(logits / temperature)."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray) -> float:
        """Find positive temperature minimizing validation negative log-likelihood."""
        logits_t = torch.tensor(val_logits, dtype=torch.float32)
        labels_t = torch.tensor(val_labels, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

        # Coarse-to-fine grid search over T > 0 (simple and deterministic).
        best_t = 1.0
        best_nll = float("inf")
        for t in np.linspace(0.25, 5.0, 96):
            with torch.no_grad():
                nll = float(criterion(logits_t / float(t), labels_t).item())
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)

        lo = max(0.05, best_t - 0.1)
        hi = best_t + 0.1
        for t in np.linspace(lo, hi, 41):
            with torch.no_grad():
                nll = float(criterion(logits_t / float(t), labels_t).item())
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)

        self.temperature = best_t
        return self.temperature

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Convert raw logits to calibrated probabilities."""
        scaled = logits / self.temperature
        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return (exp_scores / np.sum(exp_scores, axis=1, keepdims=True)).astype(np.float32)


def predict_logits(
    model: DigitMLP, images: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    """Run forward pass on numpy (N, 28, 28) images and return (N, 9) logits."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.tensor(images[start : start + batch_size], dtype=torch.float32)
            logits = model(batch).cpu().numpy()
            all_logits.append(logits)
    return np.concatenate(all_logits, axis=0)


def predict_probs(
    model: DigitMLP,
    images: np.ndarray,
    temperature: float = 1.0,
    batch_size: int = 512,
) -> np.ndarray:
    """Return calibrated class probabilities (N, 9) for numpy (N, 28, 28) images."""
    logits = predict_logits(model, images, batch_size=batch_size)
    return TemperatureScaler(temperature).apply(logits)


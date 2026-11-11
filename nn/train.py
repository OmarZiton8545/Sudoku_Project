"""Train DigitMLP with temperature scaling, evaluate vs Logistic Regression, and plot."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from data.labels import DIGITS, NUM_CLASSES, index_to_digit
from data.mnist_loader import load_processed_mnist
from nn.model import DigitMLP, TemperatureScaler, predict_logits


def set_seed(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_ece(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict]]:
    """Compute Expected Calibration Error (ECE) and per-bin stats."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == targets).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_info = []
    n = len(targets)

    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        count = int(np.sum(mask))
        if count > 0:
            acc = float(np.mean(correct[mask]))
            conf = float(np.mean(confidences[mask]))
            ece += (count / n) * abs(acc - conf)
            bins_info.append({"bin_center": 0.5 * (lo + hi), "acc": acc, "conf": conf, "count": count})
        else:
            bins_info.append({"bin_center": 0.5 * (lo + hi), "acc": None, "conf": None, "count": 0})

    return float(ece), bins_info


def compute_nll(probs: np.ndarray, targets: np.ndarray) -> float:
    """Compute mean negative log-likelihood from predicted class probabilities."""
    clipped = np.clip(probs[np.arange(len(targets)), targets], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def evaluate_probs(probs: np.ndarray, targets: np.ndarray) -> dict:
    """Return accuracy, NLL, and ECE for predicted probabilities."""
    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == targets))
    nll = compute_nll(probs, targets)
    ece, bins = compute_ece(probs, targets)
    return {"accuracy": acc, "nll": nll, "ece": ece, "bins": bins}


def _plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    """Save a 9x9 confusion matrix plot labeled with digits 1-9."""
    digit_labels = [str(index_to_digit(i)) for i in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(NUM_CLASSES), labels=digit_labels)
    ax.set_yticks(range(NUM_CLASSES), labels=digit_labels)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title("DigitMLP Test Confusion Matrix (Digits 1-9)")

    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_calibration(
    bins_before: list[dict],
    bins_after: list[dict],
    bins_logreg: list[dict],
    ece_before: float,
    ece_after: float,
    ece_logreg: float,
    out_path: Path,
) -> None:
    """Plot reliability diagram comparing NN before/after scaling and Logistic Regression."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for name, bins, ece, marker in [
        ("NN (before temp scaling)", bins_before, ece_before, "o"),
        ("NN (after temp scaling)", bins_after, ece_after, "s"),
        ("Logistic Regression", bins_logreg, ece_logreg, "^"),
    ]:
        xs = [b["conf"] for b in bins if b["count"] > 0]
        ys = [b["acc"] for b in bins if b["count"] > 0]
        ax.plot(xs, ys, marker=marker, linewidth=1.8, label=f"{name} (ECE={ece:.4f})")

    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Test Reliability Diagram (Digits 1-9)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_and_evaluate(
    seed: int = 42,
    processed_dir: str | Path = "data/processed",
    checkpoint_dir: str | Path = "nn/checkpoints",
    results_dir: str | Path = "results",
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> dict:
    """Train DigitMLP + temperature scaling, compare with LogisticRegression, save artifacts."""
    set_seed(seed)
    data = load_processed_mnist(processed_dir)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model = DigitMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(bx)
        val_probs_ep = TemperatureScaler(1.0).apply(predict_logits(model, x_val))
        val_acc = float(np.mean(np.argmax(val_probs_ep, axis=1) == y_val))
        history.append({"epoch": epoch, "train_loss": total_loss / len(x_train), "val_acc": val_acc})

    # Fit temperature scaling on validation logits.
    val_logits = predict_logits(model, x_val)
    val_before = evaluate_probs(TemperatureScaler(1.0).apply(val_logits), y_val)

    scaler = TemperatureScaler()
    temperature = scaler.fit(val_logits, y_val)
    val_after = evaluate_probs(scaler.apply(val_logits), y_val)

    # Evaluate on held-out test set before and after temperature scaling.
    test_logits = predict_logits(model, x_test)
    probs_test_before = TemperatureScaler(1.0).apply(test_logits)
    probs_test_after = scaler.apply(test_logits)
    test_before = evaluate_probs(probs_test_before, y_test)
    test_after = evaluate_probs(probs_test_after, y_test)

    # Baseline: Logistic Regression on flattened pixels.
    logreg = LogisticRegression(max_iter=500, random_state=seed)
    logreg.fit(x_train.reshape(len(x_train), -1), y_train)
    probs_logreg = logreg.predict_proba(x_test.reshape(len(x_test), -1))
    test_logreg = evaluate_probs(probs_logreg, y_test)

    preds_test = np.argmax(probs_test_after, axis=1)
    cm = confusion_matrix(y_test, preds_test, labels=list(range(NUM_CLASSES)))

    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model_file = ckpt_path / "digit_mlp.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "temperature": temperature,
            "seed": seed,
            "epochs": epochs,
        },
        model_file,
    )

    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)
    _plot_confusion_matrix(cm, res_path / "nn_confusion_matrix.png")
    _plot_calibration(
        test_before["bins"],
        test_after["bins"],
        test_logreg["bins"],
        test_before["ece"],
        test_after["ece"],
        test_logreg["ece"],
        res_path / "nn_calibration.png",
    )

    metrics = {
        "seed": seed,
        "epochs": epochs,
        "temperature": temperature,
        "digits": list(DIGITS),
        "val_before_scaling": {k: val_before[k] for k in ("accuracy", "nll", "ece")},
        "val_after_scaling": {k: val_after[k] for k in ("accuracy", "nll", "ece")},
        "test_nn_before_scaling": {k: test_before[k] for k in ("accuracy", "nll", "ece")},
        "test_nn_after_scaling": {k: test_after[k] for k in ("accuracy", "nll", "ece")},
        "test_logistic_regression": {k: test_logreg[k] for k in ("accuracy", "nll", "ece")},
        "test_misclassified_count": int(np.sum(preds_test != y_test)),
        "test_total": int(len(y_test)),
        "confusion_matrix": cm.tolist(),
        "history": history,
    }
    (res_path / "nn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_trained_model(
    checkpoint_path: str | Path = "nn/checkpoints/digit_mlp.pt",
) -> tuple[DigitMLP, float]:
    """Load trained DigitMLP weights and calibrated temperature."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = DigitMLP()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, float(ckpt["temperature"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DigitMLP and compare to Logistic Regression.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    metrics = train_and_evaluate(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(f"Temperature fitted on val: {metrics['temperature']:.4f}")
    print(
        f"Test NN (before scaling): acc={metrics['test_nn_before_scaling']['accuracy']:.4f}, "
        f"NLL={metrics['test_nn_before_scaling']['nll']:.4f}, "
        f"ECE={metrics['test_nn_before_scaling']['ece']:.4f}"
    )
    print(
        f"Test NN (after scaling) : acc={metrics['test_nn_after_scaling']['accuracy']:.4f}, "
        f"NLL={metrics['test_nn_after_scaling']['nll']:.4f}, "
        f"ECE={metrics['test_nn_after_scaling']['ece']:.4f}"
    )
    print(
        f"Test LogisticRegression : acc={metrics['test_logistic_regression']['accuracy']:.4f}, "
        f"NLL={metrics['test_logistic_regression']['nll']:.4f}, "
        f"ECE={metrics['test_logistic_regression']['ece']:.4f}"
    )
    print(
        f"Test misclassified digits: {metrics['test_misclassified_count']} / {metrics['test_total']}"
    )


if __name__ == "__main__":
    main()


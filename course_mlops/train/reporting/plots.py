import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    title: str = "Learning Curve",
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 6))

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="blue",
    )
    ax.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.1,
        color="orange",
    )

    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training score")
    ax.plot(train_sizes, val_mean, "o-", color="orange", label="Validation score")

    ax.set_xlabel("Training samples")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    display_labels = labels or ["ham", "spam"]
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=display_labels,
        cmap="Blues",
        ax=ax,
    )

    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    RocCurveDisplay.from_predictions(
        y_true,
        y_proba,
        ax=ax,
        name="Spam Classifier",
    )

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    return fig

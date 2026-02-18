import numpy as np
from matplotlib.figure import Figure

from course_mlops.train.reporting.plots import plot_confusion_matrix
from course_mlops.train.reporting.plots import plot_learning_curve
from course_mlops.train.reporting.plots import plot_roc_curve


def test_plot_learning_curve_returns_figure() -> None:
    train_sizes = np.array([10, 20, 30])
    train_scores = np.array([[0.8, 0.82], [0.85, 0.87], [0.9, 0.91]])
    val_scores = np.array([[0.7, 0.72], [0.75, 0.77], [0.8, 0.81]])
    assert isinstance(plot_learning_curve(train_sizes, train_scores, val_scores), Figure)


def test_plot_learning_curve_title() -> None:
    train_sizes = np.array([10, 20])
    train_scores = np.array([[0.8, 0.82], [0.85, 0.87]])
    val_scores = np.array([[0.7, 0.72], [0.75, 0.77]])
    fig = plot_learning_curve(train_sizes, train_scores, val_scores, title="My Curve")
    assert fig.axes[0].get_title() == "My Curve"


def test_plot_confusion_matrix_returns_figure() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    assert isinstance(plot_confusion_matrix(y_true, y_pred), Figure)


def test_plot_roc_curve_returns_figure() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    assert isinstance(plot_roc_curve(y_true, y_proba), Figure)

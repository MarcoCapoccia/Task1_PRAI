from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

FIGSIZE_DEFAULT: Tuple[float, float] = (6.0, 4.0)
FIGSIZE_CONF_MATRIX: Tuple[float, float] = (6.0, 5.0)
DPI_DEFAULT: int = 150
FONT_SIZE_BASE: int = 13


# prepares parent folders for figure output
def _prepare_path(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# applies a consistent plot style
def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI_DEFAULT,
            "font.size": FONT_SIZE_BASE,
            "axes.titlesize": FONT_SIZE_BASE + 1,
            "axes.labelsize": FONT_SIZE_BASE,
            "xtick.labelsize": FONT_SIZE_BASE - 1,
            "ytick.labelsize": FONT_SIZE_BASE - 1,
            "legend.fontsize": FONT_SIZE_BASE - 1,
        }
    )


# plots class counts for a dataset
def plot_class_distribution(
    labels: Sequence[int],
    class_names: List[str],
    path: str | Path,
    title: str,
) -> None:
    counts = np.bincount(labels)
    x = np.arange(len(class_names))

    _prepare_path(path)
    _apply_style()
    plt.figure(figsize=FIGSIZE_DEFAULT)
    bars = plt.bar(x, counts)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylabel("Number of samples")
    plt.title(title)

    # annotates each bar with the count value
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(count)}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(path, dpi=DPI_DEFAULT)
    plt.close()


# plots a simple line chart for sweeps
def plot_line(
    xs: Sequence[float],
    ys: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    path: str | Path,
) -> None:
    _prepare_path(path)
    _apply_style()
    plt.figure(figsize=FIGSIZE_DEFAULT)
    plt.plot(xs, ys, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI_DEFAULT)
    plt.close()


# plots a 2d projection by class label
def plot_scatter_2d(
    X_2d: np.ndarray,
    labels: Sequence[int],
    class_names: List[str],
    title: str,
    path: str | Path,
) -> None:
    _prepare_path(path)
    _apply_style()
    plt.figure(figsize=FIGSIZE_CONF_MATRIX)
    labels_arr = np.asarray(labels)
    # draws each class as a separate scatter layer
    for idx, name in enumerate(class_names):
        mask = labels_arr == idx
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            label=name,
            s=18,
            alpha=0.6,
        )
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI_DEFAULT)
    plt.close()


# plots a confusion matrix heatmap
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    path: str | Path,
) -> None:
    _prepare_path(path)
    _apply_style()
    plt.figure(figsize=FIGSIZE_CONF_MATRIX)
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    # annotates each confusion cell value
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.savefig(path, dpi=DPI_DEFAULT)
    plt.close()
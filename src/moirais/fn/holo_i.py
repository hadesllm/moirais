# moirais.fn — function file (hadesllm/moirais)
"""ROC curve visualization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def holo_roc(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    ax: Any | None = None,
) -> Any:
    """
    Receiver Operating Characteristic (ROC) curve with AUC annotation.

    Computes the ROC curve from scratch (no sklearn dependency) using
    threshold sweeping, then annotates the area under the curve via the
    trapezoidal rule.

    :param y_true: Binary ground-truth labels (0 or 1).
    :param y_score: Predicted scores / probabilities.
    :param ax: Optional matplotlib Axes.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    Fawcett, T. (2006). An introduction to ROC analysis. *Pattern
        Recognition Letters*, 27(8), 861--874.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_roc requires matplotlib. Install via: pip install matplotlib")
        return None

    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)

    # Sort by descending score
    order = np.argsort(-ys)
    yt = yt[order]
    ys = ys[order]

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    pos = np.sum(yt == 1)
    neg = np.sum(yt == 0)

    for i in range(len(yt)):
        if yt[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / max(pos, 1))
        fpr_list.append(fp / max(neg, 1))

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    auc = float(np.trapezoid(tpr, fpr))

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    return fig


def cheatsheet() -> str:
    return "holo_roc({}) -> ROC curve visualization."

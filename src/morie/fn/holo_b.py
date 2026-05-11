# morie.fn — function file (hadesllm/morie)
"""Box plot visualization."""

from __future__ import annotations

from typing import Any

import numpy as np


def holo_box(
    data: Any,
    col: str,
    *,
    group: str | None = None,
    ax: Any | None = None,
) -> Any:
    """
    Box plot of a numeric column, optionally grouped.

    :param data: DataFrame or array-like.
    :param col: Column to plot.
    :param group: Optional grouping column. If provided, one box per
        unique value.
    :param ax: Optional matplotlib Axes.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    McGill, R., Tukey, J. W. & Larsen, W. A. (1978). Variations of
        box plots. *The American Statistician*, 32(1), 12--16.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_box requires matplotlib. Install via: pip install matplotlib")
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if group is not None and hasattr(data, "columns") and group in data.columns:
        labels_vals = []
        tick_labels = []
        for g in sorted(data[group].unique()):
            labels_vals.append(np.asarray(data.loc[data[group] == g, col], dtype=float))
            tick_labels.append(str(g))
        ax.boxplot(labels_vals, tick_labels=tick_labels)
        ax.set_xlabel(group)
    else:
        vals = np.asarray(data[col] if hasattr(data, "__getitem__") and hasattr(data, "columns") else data, dtype=float)
        ax.boxplot(vals)

    ax.set_ylabel(col)
    ax.set_title(f"Box plot of {col}")
    return fig


def cheatsheet() -> str:
    return "holo_box({}) -> Box plot visualization."

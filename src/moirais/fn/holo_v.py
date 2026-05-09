# moirais.fn — function file (hadesllm/moirais)
"""Violin plot visualization."""

from __future__ import annotations

from typing import Any

import numpy as np


def holo_violin(
    data: Any,
    col: str,
    *,
    group: str | None = None,
    ax: Any | None = None,
) -> Any:
    """
    Violin plot of a numeric column, optionally grouped.

    :param data: DataFrame or array-like.
    :param col: Column to plot.
    :param group: Optional grouping column.
    :param ax: Optional matplotlib Axes.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    Hintze, J. L. & Nelson, R. D. (1998). Violin plots: a box
        plot-density trace synergism. *The American Statistician*,
        52(2), 181--184.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_violin requires matplotlib. Install via: pip install matplotlib")
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if group is not None and hasattr(data, "columns") and group in data.columns:
        groups_data = []
        tick_labels = []
        for g in sorted(data[group].unique()):
            groups_data.append(np.asarray(data.loc[data[group] == g, col], dtype=float))
            tick_labels.append(str(g))
        parts = ax.violinplot(groups_data, showmedians=True)
        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel(group)
    else:
        vals = np.asarray(data[col] if hasattr(data, "columns") else data, dtype=float)
        ax.violinplot([vals], showmedians=True)

    ax.set_ylabel(col)
    ax.set_title(f"Violin plot of {col}")
    return fig


def cheatsheet() -> str:
    return "holo_violin({}) -> Violin plot visualization."

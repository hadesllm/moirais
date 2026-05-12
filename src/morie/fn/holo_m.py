# morie.fn -- function file (hadesllm/morie)
"""Mosaic plot visualization."""

from __future__ import annotations

from typing import Any

import numpy as np


def holo_mosaic(
    data: Any,
    row_col: str,
    col_col: str,
    *,
    ax: Any | None = None,
) -> Any:
    """
    Mosaic plot of two categorical variables.

    Draws stacked proportional rectangles where width encodes the
    marginal frequency of *row_col* categories and height encodes the
    conditional frequency of *col_col* within each row category.

    :param data: DataFrame with categorical columns.
    :param row_col: Column for the horizontal (width) dimension.
    :param col_col: Column for the vertical (height) dimension.
    :param ax: Optional matplotlib Axes.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    Friendly, M. (1994). Mosaic displays for multi-way contingency
        tables. *Journal of the American Statistical Association*,
        89(425), 190--200.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_mosaic requires matplotlib. Install via: pip install matplotlib")
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    rows = sorted(data[row_col].unique())
    cols = sorted(data[col_col].unique())
    n = len(data)
    cmap = plt.cm.Set2

    x_offset = 0.0
    for r in rows:
        sub = data[data[row_col] == r]
        w = len(sub) / n
        y_offset = 0.0
        for ci, c in enumerate(cols):
            h = np.sum(sub[col_col] == c) / max(len(sub), 1)
            rect = mpatches.Rectangle(
                (x_offset, y_offset),
                w,
                h,
                facecolor=cmap(ci / max(len(cols) - 1, 1)),
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_patch(rect)
            if h > 0.05:
                ax.text(
                    x_offset + w / 2,
                    y_offset + h / 2,
                    str(c),
                    ha="center",
                    va="center",
                    fontsize=7,
                )
            y_offset += h
        ax.text(x_offset + w / 2, -0.04, str(r), ha="center", fontsize=8)
        x_offset += w

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(row_col)
    ax.set_ylabel(col_col)
    ax.set_title("Mosaic plot")
    return fig


def cheatsheet() -> str:
    return "holo_mosaic({}) -> Mosaic plot visualization."

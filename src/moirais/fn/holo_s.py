# moirais.fn — function file (hadesllm/moirais)
"""Scatter plot visualization."""

from __future__ import annotations

from typing import Any

import numpy as np


def holo_scatter(
    data: Any,
    x: str,
    y: str,
    *,
    hue: str | None = None,
    ax: Any | None = None,
) -> Any:
    """
    Scatter plot of two numeric columns.

    :param data: DataFrame with columns *x* and *y*.
    :param x: Column name for the horizontal axis.
    :param y: Column name for the vertical axis.
    :param hue: Optional column for colour-coding points.
    :param ax: Optional matplotlib Axes. If None a new figure is created.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_scatter requires matplotlib. Install via: pip install matplotlib")
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    xv = np.asarray(data[x], dtype=float)
    yv = np.asarray(data[y], dtype=float)

    if hue is not None and hue in getattr(data, "columns", []):
        groups = data[hue]
        for g in sorted(set(groups)):
            mask = groups == g
            ax.scatter(xv[mask], yv[mask], label=str(g), alpha=0.7)
        ax.legend(title=hue)
    else:
        ax.scatter(xv, yv, alpha=0.7)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    return fig


def cheatsheet() -> str:
    return "holo_scatter({}) -> Scatter plot visualization."

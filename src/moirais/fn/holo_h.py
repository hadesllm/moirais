# moirais.fn — function file (hadesllm/moirais)
"""Histogram visualization."""

from __future__ import annotations

from typing import Any

import numpy as np


def holo_hist(
    data: Any,
    col: str,
    *,
    bins: int = 30,
    ax: Any | None = None,
) -> Any:
    """
    Histogram of a single numeric column.

    Draws a frequency histogram using matplotlib. If *data* is a
    DataFrame the column *col* is extracted; if it is array-like it is
    used directly (and *col* is used only as the axis label).

    :param data: DataFrame or array-like of numeric values.
    :param col: Column name to plot (or label if *data* is array-like).
    :param bins: Number of bins. Default 30.
    :param ax: Optional matplotlib Axes to draw on. If None a new figure
        is created.
    :return: ``matplotlib.figure.Figure`` or ``None`` if matplotlib is
        unavailable.

    References
    ----------
    Freedman, D. & Diaconis, P. (1981). On the histogram as a density
        estimator: L2 theory. *Zeitschrift fur Wahrscheinlichkeitstheorie
        und verwandte Gebiete*, 57(4), 453--476.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_hist requires matplotlib. Install via: pip install matplotlib")
        return None

    vals = _extract(data, col)
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.hist(vals, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {col}")
    return fig


def _extract(data: Any, col: str) -> np.ndarray:
    """Extract numeric array from DataFrame, dict, or array-like."""
    if hasattr(data, "columns"):
        return np.asarray(data[col], dtype=float)
    if isinstance(data, dict) and col in data:
        return np.asarray(data[col], dtype=float)
    return np.asarray(data, dtype=float)


def cheatsheet() -> str:
    return "holo_hist({}) -> Histogram visualization."

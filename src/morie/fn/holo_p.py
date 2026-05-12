# morie.fn -- function file (hadesllm/morie)
"""Pair plot (scatter matrix) visualization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def holo_pair(
    data: Any,
    *,
    cols: Sequence[str] | None = None,
) -> Any:
    """
    Pair plot: grid of bivariate scatter plots with histograms on the
    diagonal.

    .. note::

       This function always creates its own figure (the *ax* pattern
       does not apply because multiple subplots are required).

    :param data: DataFrame with numeric columns.
    :param cols: Subset of column names to include. If None all numeric
        columns are used.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    Emerson, J. W., Green, W. A., Schloerke, B. et al. (2013). The
        generalized pairs plot. *Journal of Computational and Graphical
        Statistics*, 22(1), 79--91.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_pair requires matplotlib. Install via: pip install matplotlib")
        return None

    if cols is None:
        if hasattr(data, "select_dtypes"):
            cols = list(data.select_dtypes(include="number").columns)
        else:
            raise ValueError("cols must be specified when data is not a DataFrame")

    k = len(cols)
    fig, axes = plt.subplots(k, k, figsize=(2.5 * k, 2.5 * k))
    if k == 1:
        axes = np.array([[axes]])

    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axes[i][j]
            xi = np.asarray(data[ci], dtype=float)
            xj = np.asarray(data[cj], dtype=float)
            if i == j:
                ax.hist(xi, bins=20, edgecolor="black", alpha=0.7)
            else:
                ax.scatter(xj, xi, alpha=0.4, s=8)
            if i == k - 1:
                ax.set_xlabel(cj, fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(ci, fontsize=8)
            else:
                ax.set_yticklabels([])

    fig.suptitle("Pair plot", y=1.01)
    fig.tight_layout()
    return fig


def cheatsheet() -> str:
    return "holo_pair({}) -> Pair plot (scatter matrix) visualization."

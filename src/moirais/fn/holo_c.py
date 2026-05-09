# moirais.fn — function file (hadesllm/moirais)
"""Correlation heatmap visualization."""

from __future__ import annotations

from typing import Any

import numpy as np


def holo_corr(
    data: Any,
    *,
    method: str = "pearson",
    ax: Any | None = None,
) -> Any:
    """
    Correlation heatmap of numeric columns in a DataFrame.

    Computes the pairwise correlation matrix and displays it as a
    colour-coded heatmap with annotated coefficients.

    :param data: DataFrame with numeric columns.
    :param method: Correlation method (``"pearson"``, ``"spearman"``,
        or ``"kendall"``).  Default ``"pearson"``.
    :param ax: Optional matplotlib Axes.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    Friendly, M. (2002). Corrgrams: exploratory displays for correlation
        matrices. *The American Statistician*, 56(4), 316--324.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_corr requires matplotlib. Install via: pip install matplotlib")
        return None

    if hasattr(data, "corr"):
        corr = data.corr(method=method)
        labels = list(corr.columns)
        mat = corr.values
    else:
        mat = np.corrcoef(np.asarray(data, dtype=float).T)
        labels = [str(i) for i in range(mat.shape[0])]

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(f"Correlation heatmap ({method})")
    if fig is not None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def cheatsheet() -> str:
    return "holo_corr({}) -> Correlation heatmap visualization."

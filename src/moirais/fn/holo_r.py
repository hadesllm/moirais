# moirais.fn — function file (hadesllm/moirais)
"""Residual plot visualization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def holo_resid(
    fitted: Sequence[float],
    residuals: Sequence[float],
    *,
    ax: Any | None = None,
) -> Any:
    """
    Residual-vs-fitted-values plot.

    Scatter of residuals against fitted values with a horizontal
    reference line at zero.  Useful for diagnosing heteroscedasticity
    and non-linearity in regression models.

    :param fitted: Fitted (predicted) values.
    :param residuals: Residuals (observed - fitted).
    :param ax: Optional matplotlib Axes.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    Anscombe, F. J. (1973). Graphs in statistical analysis. *The
        American Statistician*, 27(1), 17--21.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_resid requires matplotlib. Install via: pip install matplotlib")
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    f = np.asarray(fitted, dtype=float)
    r = np.asarray(residuals, dtype=float)
    ax.scatter(f, r, alpha=0.6, edgecolors="black", linewidths=0.3)
    ax.axhline(0, linestyle="--", color="red", linewidth=0.8)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual plot")
    return fig


def cheatsheet() -> str:
    return "holo_resid({}) -> Residual plot visualization."

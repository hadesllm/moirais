# moirais.fn — function file (hadesllm/moirais)
"""QQ plot visualization."""

from __future__ import annotations

from typing import Any

import numpy as np


def holo_qq(
    data: Any,
    col: str,
    *,
    dist: str = "norm",
    ax: Any | None = None,
) -> Any:
    """
    Quantile-quantile plot against a theoretical distribution.

    Uses ``scipy.stats.probplot`` for quantile computation.

    :param data: DataFrame or array-like.
    :param col: Column name (or label if array-like).
    :param dist: Distribution name recognised by ``scipy.stats``.
        Default ``"norm"``.
    :param ax: Optional matplotlib Axes.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    Wilk, M. B. & Gnanadesikan, R. (1968). Probability plotting methods
        for the analysis of data. *Biometrika*, 55(1), 1--17.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_qq requires matplotlib. Install via: pip install matplotlib")
        return None
    try:
        from scipy import stats as sp_stats
    except ImportError:
        print("holo_qq requires scipy. Install via: pip install scipy")
        return None

    if hasattr(data, "columns") or isinstance(data, dict) and col in data:
        vals = np.asarray(data[col], dtype=float)
    else:
        vals = np.asarray(data, dtype=float)

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    sp_stats.probplot(vals, dist=dist, plot=ax)
    ax.set_title(f"QQ plot of {col} vs {dist}")
    return fig


def cheatsheet() -> str:
    return "holo_qq({}) -> QQ plot visualization."

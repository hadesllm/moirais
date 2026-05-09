# moirais.fn — function file (hadesllm/moirais)
"""Funnel plot visualization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def holo_funnel(
    effects: Sequence[float],
    ses: Sequence[float],
    *,
    ax: Any | None = None,
) -> Any:
    """
    Funnel plot of effect sizes against standard errors.

    Used to assess publication bias in meta-analyses.  The inverted
    funnel shows the pooled effect estimate with pseudo-95 % CI lines.

    :param effects: Point estimates (one per study).
    :param ses: Standard errors.
    :param ax: Optional matplotlib Axes.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    Sterne, J. A. C. & Egger, M. (2001). Funnel plots for detecting
        bias in meta-analysis. *Journal of Clinical Epidemiology*,
        54(10), 1046--1055.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_funnel requires matplotlib. Install via: pip install matplotlib")
        return None

    eff = np.asarray(effects, dtype=float)
    se = np.asarray(ses, dtype=float)
    pooled = float(np.average(eff, weights=1.0 / (se**2 + 1e-12)))

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.scatter(eff, se, edgecolors="black", facecolors="white", zorder=3)

    se_range = np.linspace(0, max(se) * 1.1, 100)
    ax.plot(
        pooled + 1.96 * se_range,
        se_range,
        linestyle="--",
        color="grey",
        linewidth=0.8,
    )
    ax.plot(
        pooled - 1.96 * se_range,
        se_range,
        linestyle="--",
        color="grey",
        linewidth=0.8,
    )
    ax.axvline(pooled, linestyle="-", color="black", linewidth=0.8)

    ax.set_xlabel("Effect size")
    ax.set_ylabel("Standard error")
    ax.set_title("Funnel plot")
    ax.invert_yaxis()
    return fig


def cheatsheet() -> str:
    return "holo_funnel({}) -> Funnel plot visualization."

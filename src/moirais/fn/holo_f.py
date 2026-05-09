# moirais.fn — function file (hadesllm/moirais)
"""Forest plot for meta-analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def holo_forest(
    effects: Sequence[float],
    ses: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    ax: Any | None = None,
) -> Any:
    """
    Forest plot of effect sizes with 95 % confidence interval whiskers.

    Each study is drawn as a point estimate with horizontal error bars
    spanning ``effect +/- 1.96 * SE``.

    :param effects: Point estimates (one per study).
    :param ses: Standard errors (same length as *effects*).
    :param labels: Optional study labels for the y-axis.
    :param ax: Optional matplotlib Axes.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    Lewis, S. & Clarke, M. (2001). Forest plots: trying to see the wood
        and the trees. *BMJ*, 322(7300), 1479--1480.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_forest requires matplotlib. Install via: pip install matplotlib")
        return None

    eff = np.asarray(effects, dtype=float)
    se = np.asarray(ses, dtype=float)
    n = len(eff)
    if labels is None:
        labels = [f"Study {i + 1}" for i in range(n)]

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    y_pos = np.arange(n)
    ci = 1.96 * se
    ax.errorbar(eff, y_pos, xerr=ci, fmt="o", color="black", capsize=3)
    ax.axvline(0, linestyle="--", color="grey", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Effect size")
    ax.set_title("Forest plot")
    ax.invert_yaxis()
    return fig


def cheatsheet() -> str:
    return "holo_forest({}) -> Forest plot for meta-analysis."

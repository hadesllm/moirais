# morie.fn -- function file (hadesllm/morie)
"""Effect size forest plot visualization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def holo_effect(
    estimates: Sequence[float],
    cis: Sequence[tuple[float, float]],
    *,
    labels: Sequence[str] | None = None,
    ax: Any | None = None,
) -> Any:
    """
    Effect size forest plot with confidence intervals.

    Similar to the meta-analytic forest plot but accepts pre-computed
    confidence interval bounds rather than standard errors.

    :param estimates: Point estimates.
    :param cis: Sequence of ``(lower, upper)`` CI tuples.
    :param labels: Optional labels for each estimate.
    :param ax: Optional matplotlib Axes.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    Cumming, G. (2012). *Understanding the New Statistics: Effect Sizes,
        Confidence Intervals, and Meta-Analysis*. Routledge.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_effect requires matplotlib. Install via: pip install matplotlib")
        return None

    est = np.asarray(estimates, dtype=float)
    n = len(est)
    if labels is None:
        labels = [f"Estimate {i + 1}" for i in range(n)]

    lo = np.array([c[0] for c in cis], dtype=float)
    hi = np.array([c[1] for c in cis], dtype=float)

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    y_pos = np.arange(n)
    xerr_lo = est - lo
    xerr_hi = hi - est
    ax.errorbar(est, y_pos, xerr=[xerr_lo, xerr_hi], fmt="s", color="navy", capsize=3)
    ax.axvline(0, linestyle="--", color="grey", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Effect size")
    ax.set_title("Effect size plot")
    ax.invert_yaxis()
    return fig


def cheatsheet() -> str:
    return "holo_effect({}) -> Effect size forest plot visualization."

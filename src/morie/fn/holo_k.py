# morie.fn — function file (hadesllm/morie)
"""Kaplan-Meier survival curve."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def holo_km(
    times: Sequence[float],
    events: Sequence[int],
    *,
    groups: Sequence[Any] | None = None,
    ax: Any | None = None,
) -> Any:
    """
    Kaplan-Meier survival curve with optional group stratification.

    Draws a step function for the estimated survival probability over
    time.  When *groups* is provided, one curve per unique group value
    is drawn.

    :param times: Observed times (durations).
    :param events: Event indicator (1 = event, 0 = censored).
    :param groups: Optional grouping variable (same length as *times*).
    :param ax: Optional matplotlib Axes.
    :return: ``matplotlib.figure.Figure`` or ``None``.

    References
    ----------
    Kaplan, E. L. & Meier, P. (1958). Nonparametric estimation from
        incomplete observations. *Journal of the American Statistical
        Association*, 53(282), 457--481.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("holo_km requires matplotlib. Install via: pip install matplotlib")
        return None

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)

    if groups is not None:
        g_arr = np.asarray(groups)
        for g in sorted(set(g_arr)):
            mask = g_arr == g
            _draw_km(ax, t[mask], e[mask], label=str(g))
        ax.legend(title="Group")
    else:
        _draw_km(ax, t, e)

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_title("Kaplan-Meier curve")
    ax.set_ylim(-0.05, 1.05)
    return fig


def _draw_km(ax: Any, t: np.ndarray, e: np.ndarray, label: str | None = None) -> None:
    """Compute and draw a single KM curve on *ax*."""
    order = np.argsort(t)
    t_s, e_s = t[order], e[order]
    unique_times = np.unique(t_s[e_s == 1])
    surv = 1.0
    times_plot = [0.0]
    surv_plot = [1.0]
    for ut in unique_times:
        at_risk = np.sum(t_s >= ut)
        events_at = np.sum((t_s == ut) & (e_s == 1))
        if at_risk > 0:
            surv *= 1.0 - events_at / at_risk
        times_plot.append(ut)
        surv_plot.append(surv)
    ax.step(times_plot, surv_plot, where="post", label=label)


def cheatsheet() -> str:
    return "holo_km({}) -> Kaplan-Meier survival curve."

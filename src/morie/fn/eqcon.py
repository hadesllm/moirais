# morie.fn — function file (hadesllm/morie)
"""Health concentration index."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def concentration_index(
    health_var: np.ndarray | list[float],
    ses_rank: np.ndarray | list[float],
) -> ESRes:
    """Compute health concentration index.

    CI = (2 / (n * mu)) * sum((R_i - 0.5) * h_i)
    where R_i is fractional rank by SES.

    Parameters
    ----------
    health_var : array-like
        Health outcome values.
    ses_rank : array-like
        Socioeconomic status ranking variable (higher = richer).

    Returns
    -------
    ESRes
    """
    h = np.asarray(health_var, dtype=float)
    s = np.asarray(ses_rank, dtype=float)
    if len(h) != len(s) or len(h) < 3:
        raise ValueError("Need at least 3 observations, same length")
    order = np.argsort(s)
    h_sorted = h[order]
    n = len(h)
    mu = np.mean(h)
    if abs(mu) < 1e-12:
        raise ValueError("Mean health variable is zero")
    ranks = (np.arange(1, n + 1) - 0.5) / n
    ci = float((2 / (n * mu)) * np.sum((ranks - 0.5) * h_sorted))
    return ESRes(measure="concentration_index", estimate=ci, n=n, extra={"mean_health": float(mu)})


eqcon = concentration_index


def cheatsheet() -> str:
    return "concentration_index({}) -> Health concentration index."

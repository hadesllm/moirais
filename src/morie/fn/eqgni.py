# morie.fn -- function file (hadesllm/morie)
"""Gini coefficient for inequality."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def gini_coefficient(
    values: np.ndarray | list[float],
) -> ESRes:
    """Compute Gini coefficient of inequality.

    G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
    where x_i are sorted values.

    Parameters
    ----------
    values : array-like
        Non-negative values (income, rates, etc.).

    Returns
    -------
    ESRes
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        raise ValueError("Need at least 2 values")
    if np.any(v < 0):
        raise ValueError("Values must be non-negative")
    v_sorted = np.sort(v)
    n = len(v_sorted)
    index = np.arange(1, n + 1)
    gini = float((2 * np.sum(index * v_sorted)) / (n * np.sum(v_sorted)) - (n + 1) / n)
    return ESRes(measure="gini_coefficient", estimate=gini, n=n)


eqgni = gini_coefficient


def cheatsheet() -> str:
    return "gini_coefficient({}) -> Gini coefficient for inequality."

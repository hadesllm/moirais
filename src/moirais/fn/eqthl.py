# moirais.fn — function file (hadesllm/moirais)
"""Theil's entropy index."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import ESRes


def theil_index(
    values: np.ndarray | list[float],
) -> ESRes:
    """Compute Theil's T (entropy) index of inequality.

    T = (1/n) * sum((x_i/mu) * ln(x_i/mu))

    Parameters
    ----------
    values : array-like
        Positive values.

    Returns
    -------
    ESRes
    """
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    if len(v) < 2:
        raise ValueError("Need at least 2 positive values")
    mu = np.mean(v)
    ratios = v / mu
    theil = float(np.mean(ratios * np.log(ratios)))
    return ESRes(measure="theil_index", estimate=theil, n=len(v))


eqthl = theil_index


def cheatsheet() -> str:
    return "theil_index({}) -> Theil's entropy index."

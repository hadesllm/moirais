# moirais.fn — function file (hadesllm/moirais)
"""Atkinson inequality index."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import ESRes


def atkinson_index(
    values: np.ndarray | list[float],
    *,
    epsilon: float = 0.5,
) -> ESRes:
    """Compute Atkinson inequality index.

    A(epsilon) = 1 - (1/mu) * (1/n * sum(x_i^(1-e)))^(1/(1-e))
    for epsilon != 1.

    Parameters
    ----------
    values : array-like
        Positive values.
    epsilon : float
        Inequality aversion parameter (0 = indifferent, >1 = more averse).

    Returns
    -------
    ESRes
    """
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    if len(v) < 2:
        raise ValueError("Need at least 2 positive values")
    mu = np.mean(v)
    n = len(v)
    if abs(epsilon - 1.0) < 1e-12:
        geometric_mean = np.exp(np.mean(np.log(v)))
        atk = 1 - geometric_mean / mu
    else:
        power = 1 - epsilon
        generalised_mean = (np.mean(v**power)) ** (1 / power)
        atk = 1 - generalised_mean / mu
    return ESRes(measure="atkinson_index", estimate=float(atk), n=n, extra={"epsilon": epsilon})


eqatn = atkinson_index


def cheatsheet() -> str:
    return "atkinson_index({}) -> Atkinson inequality index."

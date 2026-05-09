# moirais.fn — function file (hadesllm/moirais)
"""Composite reliability from factor loadings."""

from __future__ import annotations

import numpy as np


def crel(loads: np.ndarray) -> float:
    """Composite reliability from standardized factor loadings.

    CR = (sum lambda)^2 / ((sum lambda)^2 + sum(1 - lambda^2))

    Parameters
    ----------
    loads : ndarray
        Standardized factor loadings (1-D array).

    Returns
    -------
    float
        Composite reliability coefficient.
    """
    lam = np.asarray(loads, dtype=np.float64)
    sl = lam.sum()
    se = (1 - lam**2).sum()
    return float(sl**2 / (sl**2 + se))


def cheatsheet() -> str:
    return "crel({}) -> Composite reliability from factor loadings."

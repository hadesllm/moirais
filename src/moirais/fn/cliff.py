# moirais.fn — function file (hadesllm/moirais)
"""Cliff's delta (non-parametric effect size)."""

from typing import Union

import numpy as np
import pandas as pd

from ._containers import ESRes
from ._helpers import _arr, _bootstrap_ci


def cliffs_delta(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> ESRes:
    """Cliff's delta (non-parametric effect size).

    delta = (#(xi > yj) - #(xi < yj)) / (nx * ny)

    Parameters
    ----------
    x, y : array-like
    confidence : float, default 0.95

    Returns
    -------
    ESRes
    """
    x, y = _arr(x), _arr(y)
    nx, ny = len(x), len(y)
    greater = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                greater += 1
            elif xi < yj:
                less += 1
    delta = (greater - less) / (nx * ny) if nx * ny > 0 else 0.0

    def _delta(a, b):
        g = sum(1 for ai in a for bj in b if ai > bj)
        l_ = sum(1 for ai in a for bj in b if ai < bj)
        return (g - l_) / (len(a) * len(b))

    se, ci_lo, ci_hi = _bootstrap_ci(_delta, (x, y), confidence=confidence)
    return ESRes(
        measure="Cliff's delta",
        estimate=float(delta),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=nx + ny,
    )


cliff = cliffs_delta


def cheatsheet() -> str:
    return "cliffs_delta({}) -> Cliff's delta (non-parametric effect size)."

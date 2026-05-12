# morie.fn -- function file (hadesllm/morie)
"""Common Language Effect Size (probability of superiority)."""

from typing import Union

import numpy as np
import pandas as pd

from ._containers import ESRes
from ._helpers import _arr, _bootstrap_ci


def cles(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    confidence: float = 0.95,
) -> ESRes:
    """Common Language Effect Size (probability of superiority).

    Estimates P(X > Y) for randomly drawn observations from each group.

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
    count = 0
    ties = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                count += 1
            elif xi == yj:
                ties += 1
    p_sup = (count + 0.5 * ties) / (nx * ny) if nx * ny > 0 else 0.5
    se, ci_lo, ci_hi = _bootstrap_ci(
        lambda a, b: sum(1 for ai in a for bj in b if ai > bj) / (len(a) * len(b)),
        (x, y),
        confidence=confidence,
    )
    return ESRes(
        measure="CLES (Prob. of superiority)",
        estimate=float(p_sup),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=nx + ny,
    )


psup = cles


def cheatsheet() -> str:
    return "cles({}) -> Common Language Effect Size (probability of superiority)."

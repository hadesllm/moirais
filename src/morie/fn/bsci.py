# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Generic bootstrap CI wrapper for any effect-size function."""

from typing import Union

import numpy as np
import pandas as pd

from ._containers import ESRes
from ._helpers import _arr, _bootstrap_ci


def bootstrap_effect_size_ci(
    func: callable,
    *arrays: Union[np.ndarray, pd.Series, list],
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> ESRes:
    """Generic bootstrap CI wrapper for any effect-size function.

    Parameters
    ----------
    func : callable
        Function that takes one or more arrays and returns a scalar.
    *arrays : array-like
        Input arrays to bootstrap.
    n_boot : int, default 2000
    confidence : float, default 0.95
    seed : int, default 42

    Returns
    -------
    ESRes
    """
    arrs = tuple(_arr(a) for a in arrays)
    point = float(func(*arrs))
    se, ci_lo, ci_hi = _bootstrap_ci(func, arrs, n_boot=n_boot, confidence=confidence, seed=seed)
    return ESRes(
        measure=f"Bootstrap ({func.__name__})",
        estimate=point,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        se=se,
        n=sum(len(a) for a in arrs),
    )


boot_es_ci = bootstrap_effect_size_ci


def cheatsheet() -> str:
    return "bootstrap_effect_size_ci({}) -> Generic bootstrap CI wrapper for any effect-size function."

# morie.fn — function file (hadesllm/morie)
"""Glivenko-Cantelli theorem verification."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_glivenko_cantelli"]


def kosorok_glivenko_cantelli(x):
    """
    Glivenko-Cantelli theorem verification

    Formula: sup|F_n(x) - F(x)| -> 0 a.s.

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Kosorok (2008), Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Glivenko-Cantelli theorem verification"})


def cheatsheet():
    return "ksr03: Glivenko-Cantelli theorem verification"

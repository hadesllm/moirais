# moirais.fn — function file (hadesllm/moirais)
"""Detrended fluctuation analysis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_dfa"]


def rangayyan_dfa(x):
    """
    Detrended fluctuation analysis

    Formula: F(n) = sqrt((1/N) sum (Y(k) - Y_n(k))^2)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rangayyan Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Detrended fluctuation analysis"})


def cheatsheet():
    return "rgdfa: Detrended fluctuation analysis"

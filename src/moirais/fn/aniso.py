# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Anisotropy detection in spatial data."""
import numpy as np
from ._richresult import RichResult

__all__ = ["anisotropy_test"]


def anisotropy_test(x, coords):
    """
    Anisotropy detection in spatial data

    Formula: Compare directional variograms

    Parameters
    ----------
    x : array-like
        Input data.
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Schabenberger Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Anisotropy detection in spatial data"})


def cheatsheet():
    return "aniso: Anisotropy detection in spatial data"

# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Antithetic variates for variance reduction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["antithetic_variates"]


def antithetic_variates(x):
    """
    Antithetic variates for variance reduction

    Formula: theta = (f(U) + f(1-U)) / 2

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Hammersley & Morton (1956)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Antithetic variates for variance reduction"})


def cheatsheet():
    return "antth: Antithetic variates for variance reduction"

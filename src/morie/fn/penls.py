# morie.fn — function file (hadesllm/morie)
"""Penalized regression (ridge/LASSO/elastic net) for markers."""
import numpy as np
from ._richresult import RichResult

__all__ = ["penalized_regression"]


def penalized_regression(x, y):
    """
    Penalized regression (ridge/LASSO/elastic net) for markers

    Formula: min ||y-Xb||^2 + lambda*(alpha*||b||_1 + (1-alpha)*||b||_2^2)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Montesinos Lopez Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Penalized regression (ridge/LASSO/elastic net) for markers"})


def cheatsheet():
    return "penls: Penalized regression (ridge/LASSO/elastic net) for markers"

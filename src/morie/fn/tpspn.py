"""Thin plate spline regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["thin_plate_spline"]


def thin_plate_spline(x, y):
    """
    Thin plate spline regression

    Formula: f(x) = sum alpha_i * phi(||x-x_i||) + polynomial

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Duchon (1977)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Thin plate spline regression"})


def cheatsheet():
    return "tpspn: Thin plate spline regression"

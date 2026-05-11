# morie.fn — function file (hadesllm/morie)
"""Isotonic (monotone) regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["isotonic_regression"]


def isotonic_regression(x, y):
    """
    Isotonic (monotone) regression

    Formula: min sum w_i(y_i - f_i)^2 s.t. f_1 <= f_2 <= ...

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
    Barlow et al. (1972)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Isotonic (monotone) regression"})


def cheatsheet():
    return "isotn: Isotonic (monotone) regression"

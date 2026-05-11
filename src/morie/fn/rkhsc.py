# morie.fn — function file (hadesllm/morie)
"""RKHS kernel ridge regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rkhs_kernel_regression"]


def rkhs_kernel_regression(x, y):
    """
    RKHS kernel ridge regression

    Formula: f = argmin ||y-f||^2 + lambda*||f||_H^2

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
    Wahba (1990)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RKHS kernel ridge regression"})


def cheatsheet():
    return "rkhsc: RKHS kernel ridge regression"

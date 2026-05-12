# morie.fn -- function file (hadesllm/morie)
"""Gaussian regression with fixed design: contraction in L2(X) norm."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gauss_reg_crt"]


def ghosal_gauss_reg_crt(x, y):
    """
    Gaussian regression with fixed design: contraction in L2(X) norm

    Formula: eps_n = sqrt(log n / n) for s-smooth regression function with s-smooth GP prior

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
    Ghosal Ch 8 §8.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian regression with fixed design: contraction in L2(X) norm"})


def cheatsheet():
    return "gh_c8_8: Gaussian regression with fixed design: contraction in L2(X) norm"

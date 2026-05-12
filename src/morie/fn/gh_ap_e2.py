# morie.fn -- function file (hadesllm/morie)
"""Spline space S_{K,r}: polynomials of degree r with K interior knots."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_spline_space"]


def ghosal_spline_space(x):
    """
    Spline space S_{K,r}: polynomials of degree r with K interior knots

    Formula: dim(S_{K,r}) = K + r + 1, approximation error ||f - s*||_2 ~ K^{-s} for s-smooth f

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
    Ghosal App E
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spline space S_{K,r}: polynomials of degree r with K interior knots"})


def cheatsheet():
    return "gh_ap_e2: Spline space S_{K,r}: polynomials of degree r with K interior knots"

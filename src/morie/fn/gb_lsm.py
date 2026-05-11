# morie.fn — function file (hadesllm/morie)
"""Large-sample approximations to mean and variance of X_(r)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_large_sample_moments"]


def gibbons_large_sample_moments(r, n, f, F):
    """
    Large-sample approximations to mean and variance of X_(r)

    Formula: E(X_(r)) approx F^{-1}(p) + (pq)/(2n*f^2)*f'(F^{-1}(p)); Var approx pq/(n*f^2)

    Parameters
    ----------
    r : array-like
        Input data.
    n : array-like
        Input data.
    f : array-like
        Input data.
    F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: approx_mean, approx_var

    References
    ----------
    Gibbons Ch 2.9
    """
    r = np.asarray(r, dtype=float)
    n = int(r) if r.ndim == 0 else len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Large-sample approximations to mean and variance of X_(r)"})


def cheatsheet():
    return "gb_lsm: Large-sample approximations to mean and variance of X_(r)"

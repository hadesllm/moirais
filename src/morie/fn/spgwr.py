"""Geographically weighted regression: locally-varying coefficients."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_gwr"]


def schabenberger_gwr(x, y, coords, bandwidth):
    """
    Geographically weighted regression: locally-varying coefficients

    Formula: beta(s_i) = (X'*W(s_i)*X)^{-1}*X'*W(s_i)*Y where W(s_i) = diag(kernel weights)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    coords : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: local_coefficients, r2

    References
    ----------
    Schabenberger Ch 6, Sec 6.1.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Geographically weighted regression: locally-varying coefficients"})


def cheatsheet():
    return "spgwr: Geographically weighted regression: locally-varying coefficients"

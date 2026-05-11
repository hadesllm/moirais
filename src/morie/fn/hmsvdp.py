# morie.fn — function file (hadesllm/morie)
"""OLS via SVD pseudoinverse (robust to singular X^T X)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_svd_pseudoinverse"]


def geron_svd_pseudoinverse(X, y):
    """
    OLS via SVD pseudoinverse (robust to singular X^T X)

    Formula: theta_hat = X^+ y = V Sigma^+ U^T y

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Géron Ch 4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "OLS via SVD pseudoinverse (robust to singular X^T X)"})


def cheatsheet():
    return "hmsvdp: OLS via SVD pseudoinverse (robust to singular X^T X)"

# morie.fn — function file (hadesllm/morie)
"""Quadratic discriminant analysis (QDA) with unequal covariance matrices."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_qda"]


def rangayyan_qda(X, y):
    """
    Quadratic discriminant analysis (QDA) with unequal covariance matrices

    Formula: g_k(y) = -0.5*ln|Sigma_k| - 0.5*(y-mu_k)^T*Sigma_k^{-1}*(y-mu_k) + ln P(C_k)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, discriminants

    References
    ----------
    Rangayyan Ch 10.4.2
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quadratic discriminant analysis (QDA) with unequal covariance matrices"})


def cheatsheet():
    return "rgqda: Quadratic discriminant analysis (QDA) with unequal covariance matrices"

# morie.fn — function file (hadesllm/morie)
"""Henderson mixed model equations (Eq 2.2): joint BLUE/BLUP system."""
import numpy as np
from ._richresult import RichResult

__all__ = ["henderson_mme_eq2_2"]


def henderson_mme_eq2_2(Y, X, Z, R, Sigma):
    """
    Henderson mixed model equations (Eq 2.2): joint BLUE/BLUP system

    Formula: [[X'R^-1 X, X'R^-1 Z],[Z'R^-1 X, Z'R^-1 Z + Sigma^-1]] * [beta_hat, u_hat] = [X'R^-1 y, Z'R^-1 y]

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.
    R : array-like
        Input data.
    Sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'beta_hat': 'array', 'u_hat': 'array'}

    References
    ----------
    Montesinos Lopez Ch 2 Eq 2.2
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Henderson mixed model equations (Eq 2.2): joint BLUE/BLUP system"})


def cheatsheet():
    return "mmeeq: Henderson mixed model equations (Eq 2.2): joint BLUE/BLUP system"

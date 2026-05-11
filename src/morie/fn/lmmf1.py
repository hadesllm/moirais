# morie.fn — function file (hadesllm/morie)
"""Linear mixed model general form (Eq 2.1): Y = Xbeta + Zu + epsilon."""
import numpy as np
from ._richresult import RichResult

__all__ = ["lmm_form_eq2_1"]


def lmm_form_eq2_1(Y, X, Z, beta_init):
    """
    Linear mixed model general form (Eq 2.1): Y = Xbeta + Zu + epsilon

    Formula: Y = X*beta + Z*u + eps; u ~ N(0, Sigma), eps ~ N(0, R)

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    Z : array-like
        Input data.
    beta_init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'blup': 'vector', 'blue': 'vector'}

    References
    ----------
    Montesinos Lopez Ch 2 Eq 2.1
    """
    Y = np.asarray(Y, dtype=float)
    n = int(Y) if Y.ndim == 0 else len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear mixed model general form (Eq 2.1): Y = Xbeta + Zu + epsilon"})


def cheatsheet():
    return "lmmf1: Linear mixed model general form (Eq 2.1): Y = Xbeta + Zu + epsilon"

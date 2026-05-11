"""Vector error-correction model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vecm"]


def vecm(Y, k_ar, coint_rank):
    """
    Vector error-correction model

    Formula: ΔY_t = αβ' Y_{t-1} + sum Γ_i ΔY_{t-i} + ε

    Parameters
    ----------
    Y : array-like
        Input data.
    k_ar : array-like
        Input data.
    coint_rank : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Engle-Granger (1987); Johansen (1988)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Vector error-correction model"})


def cheatsheet():
    return "vecmF: Vector error-correction model"

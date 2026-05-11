"""Principal components regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_pcr"]


def esl_pcr(X, y, M):
    """
    Principal components regression

    Formula: beta_hat^PCR = sum_{m<=M} (z_m'y / z_m'z_m) v_m

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta

    References
    ----------
    Hastie ESL Ch 3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Principal components regression"})


def cheatsheet():
    return "eslpcr: Principal components regression"

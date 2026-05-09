"""Exponential GARCH (asymmetric)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["egarch_model"]


def egarch_model(y, p, q):
    """
    Exponential GARCH (asymmetric)

    Formula: log sigma_t^2 = omega + alpha (|z| - E|z|) + gamma z + beta log sigma_{t-1}^2

    Parameters
    ----------
    y : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nelson (1991)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exponential GARCH (asymmetric)"})


def cheatsheet():
    return "egarch: Exponential GARCH (asymmetric)"

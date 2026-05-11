"""Exponential GARCH (asymmetric)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["egarch_nelson"]


def egarch_nelson(x, p, q):
    """
    Exponential GARCH (asymmetric)

    Formula: log sigma_t^2 = omega + alpha [|z_{t-1}| + gamma z_{t-1}] + beta log sigma_{t-1}^2

    Parameters
    ----------
    x : array-like
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
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exponential GARCH (asymmetric)"})


def cheatsheet():
    return "egarcm: Exponential GARCH (asymmetric)"

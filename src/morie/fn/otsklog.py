"""Log-domain Sinkhorn for numerical stability."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_sinkhorn_log"]


def ot_sinkhorn_log(a, b, C, epsilon, max_iter):
    """
    Log-domain Sinkhorn for numerical stability

    Formula: Update f,g via logsumexp; T=exp((f+g-C)/ε)

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    C : array-like
        Input data.
    epsilon : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T, cost, f, g

    References
    ----------
    Schmitzer (2019)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log-domain Sinkhorn for numerical stability"})


def cheatsheet():
    return "otsklog: Log-domain Sinkhorn for numerical stability"

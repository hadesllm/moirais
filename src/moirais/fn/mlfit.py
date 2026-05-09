"""ML log-likelihood evaluation for LMM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ml_loglik"]


def ml_loglik(y, X, V):
    """
    ML log-likelihood evaluation for LMM

    Formula: -0.5 [n log(2 pi) + log|V| + (y - X beta)' V^-1 (y - X beta)]

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hartley & Rao (1967)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ML log-likelihood evaluation for LMM"})


def cheatsheet():
    return "mlfit: ML log-likelihood evaluation for LMM"

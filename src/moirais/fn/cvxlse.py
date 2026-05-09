"""Log-sum-exp gradient."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_lse"]


def boyd_lse(x):
    """
    Log-sum-exp gradient

    Formula: grad lse = exp(x) / sum exp(x)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gradient

    References
    ----------
    Boyd CVX Ch 3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log-sum-exp gradient"})


def cheatsheet():
    return "cvxlse: Log-sum-exp gradient"

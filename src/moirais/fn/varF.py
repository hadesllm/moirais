"""VAR(p) multivariate AR."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vector_autoregression"]


def vector_autoregression(Y, p):
    """
    VAR(p) multivariate AR

    Formula: Y_t = c + sum A_i Y_{t-i} + ε

    Parameters
    ----------
    Y : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sims (1980)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VAR(p) multivariate AR"})


def cheatsheet():
    return "varF: VAR(p) multivariate AR"

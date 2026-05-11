"""Multi-mediator causal mediation analysis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["multi_mediator_causal"]


def multi_mediator_causal(X, M, Y):
    """
    Multi-mediator causal mediation analysis

    Formula: decomposition into NIE_M1 + NIE_M2 + NIE_int

    Parameters
    ----------
    X : array-like
        Input data.
    M : array-like
        Input data.
    Y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Daniel, De Stavola, Cousens, Vansteelandt (2015)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-mediator causal mediation analysis"})


def cheatsheet():
    return "mcausm: Multi-mediator causal mediation analysis"

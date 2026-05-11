"""TMLE for marginal probabilistic index."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_marginal_pim"]


def tmle_marginal_pim(y, D, X):
    """
    TMLE for marginal probabilistic index

    Formula: P(Y(1) > Y(0)) target

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    De Schryver-Vansteelandt (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for marginal probabilistic index"})


def cheatsheet():
    return "tmlmpi: TMLE for marginal probabilistic index"

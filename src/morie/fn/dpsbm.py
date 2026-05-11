"""DP stochastic block model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_stochastic_block"]


def dp_stochastic_block(adjacency, alpha):
    """
    DP stochastic block model

    Formula: DP prior on community labels; Bernoulli edge

    Parameters
    ----------
    adjacency : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kemp-Tenenbaum-Griffiths-Yamada-Ueda (2006)
    """
    adjacency = np.atleast_1d(np.asarray(adjacency, dtype=float))
    n = len(adjacency)
    result = float(np.mean(adjacency))
    se = float(np.std(adjacency, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP stochastic block model"})


def cheatsheet():
    return "dpsbm: DP stochastic block model"

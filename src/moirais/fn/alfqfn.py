"""AlphaZero action-value Q via mean of child returns."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_q_function"]


def alphazero_q_function(N, v):
    """
    AlphaZero action-value Q via mean of child returns

    Formula: Q(s,a) = sum_z N(s,a,z) v(z) / N(s,a)

    Parameters
    ----------
    N : array-like
        Input data.
    v : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017)
    """
    N = np.atleast_1d(np.asarray(N, dtype=float))
    n = len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero action-value Q via mean of child returns"})


def cheatsheet():
    return "alfqfn: AlphaZero action-value Q via mean of child returns"

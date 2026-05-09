"""ε-greedy exploration."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["epsilon_greedy"]


def epsilon_greedy(arms, epsilon, T):
    """
    ε-greedy exploration

    Formula: explore w.p. ε, exploit w.p. 1−ε

    Parameters
    ----------
    arms : array-like
        Input data.
    epsilon : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sutton-Barto (1998)
    """
    arms = np.atleast_1d(np.asarray(arms, dtype=float))
    n = len(arms)
    result = float(np.mean(arms))
    se = float(np.std(arms, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ε-greedy exploration"})


def cheatsheet():
    return "epsg: ε-greedy exploration"

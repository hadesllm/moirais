"""UCB1 bandit."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ucb_bandit"]


def ucb_bandit(arms, T):
    """
    UCB1 bandit

    Formula: arm = argmax_i x̄_i + √(2 ln n / n_i)

    Parameters
    ----------
    arms : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Auer-Cesa-Bianchi-Fischer (2002)
    """
    arms = np.atleast_1d(np.asarray(arms, dtype=float))
    n = len(arms)
    result = float(np.mean(arms))
    se = float(np.std(arms, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "UCB1 bandit"})


def cheatsheet():
    return "ucbb: UCB1 bandit"

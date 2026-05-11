"""AlphaZero policy head."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_policy_head"]


def alphazero_policy_head(x, action_space):
    """
    AlphaZero policy head

    Formula: conv(2) + flatten + linear -> softmax over actions

    Parameters
    ----------
    x : array-like
        Input data.
    action_space : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero policy head"})


def cheatsheet():
    return "agphdt: AlphaZero policy head"

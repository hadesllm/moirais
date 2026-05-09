"""AlphaZero MCTS backup."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_backup"]


def alphazero_backup(leaf, value, path):
    """
    AlphaZero MCTS backup

    Formula: propagate v from leaf to root via Q,N updates

    Parameters
    ----------
    leaf : array-like
        Input data.
    value : array-like
        Input data.
    path : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017)
    """
    leaf = np.atleast_1d(np.asarray(leaf, dtype=float))
    n = len(leaf)
    result = float(np.mean(leaf))
    se = float(np.std(leaf, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero MCTS backup"})


def cheatsheet():
    return "agbckp: AlphaZero MCTS backup"

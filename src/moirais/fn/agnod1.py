"""AlphaZero MCTS node initialization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_node_init"]


def alphazero_node_init(p, action_space):
    """
    AlphaZero MCTS node initialization

    Formula: new node: P=p, N=0, W=0, Q=0

    Parameters
    ----------
    p : array-like
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
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero MCTS node initialization"})


def cheatsheet():
    return "agnod1: AlphaZero MCTS node initialization"

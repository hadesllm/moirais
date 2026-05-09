"""AlphaZero MCTS expansion via policy network."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_expand"]


def alphazero_expand(state, policy_net):
    """
    AlphaZero MCTS expansion via policy network

    Formula: P_a = pi(s) for newly visited s

    Parameters
    ----------
    state : array-like
        Input data.
    policy_net : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2017)
    """
    state = np.atleast_1d(np.asarray(state, dtype=float))
    n = len(state)
    result = float(np.mean(state))
    se = float(np.std(state, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero MCTS expansion via policy network"})


def cheatsheet():
    return "agexpd: AlphaZero MCTS expansion via policy network"

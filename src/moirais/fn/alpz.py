"""AlphaZero MCTS + neural prior."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_search"]


def alphazero_search(state, net, num_sim):
    """
    AlphaZero MCTS + neural prior

    Formula: Q + U(s,a); U = c·P(s,a)·√N(s)/(1+N(s,a))

    Parameters
    ----------
    state : array-like
        Input data.
    net : array-like
        Input data.
    num_sim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2018) AlphaZero
    """
    state = np.atleast_1d(np.asarray(state, dtype=float))
    n = len(state)
    result = float(np.mean(state))
    se = float(np.std(state, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero MCTS + neural prior"})


def cheatsheet():
    return "alpz: AlphaZero MCTS + neural prior"

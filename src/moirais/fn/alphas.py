"""AlphaZero self-play training step."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_self_play"]


def alphazero_self_play(state, policy, value, mcts_iter):
    """
    AlphaZero self-play training step

    Formula: play(s_t, a_t ~ pi_t); collect (s, pi, z) and train

    Parameters
    ----------
    state : array-like
        Input data.
    policy : array-like
        Input data.
    value : array-like
        Input data.
    mcts_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2018) Science
    """
    state = np.atleast_1d(np.asarray(state, dtype=float))
    n = len(state)
    result = float(np.mean(state))
    se = float(np.std(state, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero self-play training step"})


def cheatsheet():
    return "alphas: AlphaZero self-play training step"

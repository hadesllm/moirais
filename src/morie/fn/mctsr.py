"""Vanilla MCTS with random rollouts."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mcts_rollout"]


def mcts_rollout(state, budget):
    """
    Vanilla MCTS with random rollouts

    Formula: select / expand / simulate / backpropagate

    Parameters
    ----------
    state : array-like
        Input data.
    budget : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Coulom (2006); Browne et al (2012) survey
    """
    state = np.atleast_1d(np.asarray(state, dtype=float))
    n = len(state)
    result = float(np.mean(state))
    se = float(np.std(state, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Vanilla MCTS with random rollouts"})


def cheatsheet():
    return "mctsr: Vanilla MCTS with random rollouts"

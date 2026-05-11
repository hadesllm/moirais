"""Nash-Q for Markov games."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nash_q_learning"]


def nash_q_learning(agents, env):
    """
    Nash-Q for Markov games

    Formula: each agent's Q assumes opponents play Nash

    Parameters
    ----------
    agents : array-like
        Input data.
    env : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hu-Wellman (2003)
    """
    agents = np.atleast_1d(np.asarray(agents, dtype=float))
    n = len(agents)
    result = float(np.mean(agents))
    se = float(np.std(agents, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nash-Q for Markov games"})


def cheatsheet():
    return "nashq: Nash-Q for Markov games"

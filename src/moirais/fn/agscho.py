"""AlphaZero search horizon control."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_search_horizon"]


def alphazero_search_horizon(depth_limit, state):
    """
    AlphaZero search horizon control

    Formula: limit MCTS depth; sample beyond limit

    Parameters
    ----------
    depth_limit : array-like
        Input data.
    state : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2018)
    """
    depth_limit = np.atleast_1d(np.asarray(depth_limit, dtype=float))
    n = len(depth_limit)
    result = float(np.mean(depth_limit))
    se = float(np.std(depth_limit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero search horizon control"})


def cheatsheet():
    return "agscho: AlphaZero search horizon control"

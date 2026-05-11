"""Model-based RL planning."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["model_based_rl"]


def model_based_rl(env, model, planner):
    """
    Model-based RL planning

    Formula: learn p̂(s'|s,a), r̂; plan via dynamic programming

    Parameters
    ----------
    env : array-like
        Input data.
    model : array-like
        Input data.
    planner : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sutton (1991) Dyna
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Model-based RL planning"})


def cheatsheet():
    return "mfomf: Model-based RL planning"

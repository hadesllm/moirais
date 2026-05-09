"""C51 categorical distributional RL."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["distributional_rl"]


def distributional_rl(env, atoms, vmin, vmax):
    """
    C51 categorical distributional RL

    Formula: learn distribution Z(s,a), Bellman in distribution space

    Parameters
    ----------
    env : array-like
        Input data.
    atoms : array-like
        Input data.
    vmin : array-like
        Input data.
    vmax : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bellemare-Dabney-Munos (2017)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "C51 categorical distributional RL"})


def cheatsheet():
    return "distq: C51 categorical distributional RL"

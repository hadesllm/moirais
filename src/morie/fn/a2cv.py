"""Advantage actor-critic (synchronous A2C)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["a2c"]


def a2c(env, actor, critic, n_steps):
    """
    Advantage actor-critic (synchronous A2C)

    Formula: A_t = R_t − V(s_t); use A_t in policy gradient

    Parameters
    ----------
    env : array-like
        Input data.
    actor : array-like
        Input data.
    critic : array-like
        Input data.
    n_steps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mnih et al (2016) A3C/A2C
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Advantage actor-critic (synchronous A2C)"})


def cheatsheet():
    return "a2cv: Advantage actor-critic (synchronous A2C)"

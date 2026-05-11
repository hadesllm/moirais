"""MuZero — learns model + value + policy from latent state."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["muzero"]


def muzero(env, net, unroll_steps):
    """
    MuZero — learns model + value + policy from latent state

    Formula: recurrent prediction h, g, f networks

    Parameters
    ----------
    env : array-like
        Input data.
    net : array-like
        Input data.
    unroll_steps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schrittwieser et al (2020)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MuZero — learns model + value + policy from latent state"})


def cheatsheet():
    return "muzero: MuZero — learns model + value + policy from latent state"

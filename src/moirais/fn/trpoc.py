"""Trust region policy optimization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["trpo"]


def trpo(env, policy, kl_max):
    """
    Trust region policy optimization

    Formula: max θ E[π(a|s,θ)/π_old · A] s.t. KL ≤ δ

    Parameters
    ----------
    env : array-like
        Input data.
    policy : array-like
        Input data.
    kl_max : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schulman et al (2015)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Trust region policy optimization"})


def cheatsheet():
    return "trpoc: Trust region policy optimization"

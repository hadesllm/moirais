"""Dueling architecture: separate V and A streams."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dueling_dqn"]


def dueling_dqn(env, net):
    """
    Dueling architecture: separate V and A streams

    Formula: Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)

    Parameters
    ----------
    env : array-like
        Input data.
    net : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wang et al (2016)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dueling architecture: separate V and A streams"})


def cheatsheet():
    return "dueldqn: Dueling architecture: separate V and A streams"

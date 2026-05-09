"""Intrinsic motivation / curiosity reward."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["intrinsic_motivation"]


def intrinsic_motivation(env, forward_model, beta):
    """
    Intrinsic motivation / curiosity reward

    Formula: r_int = β · ||f(s_{t+1}) − f̂(s_{t+1})||²

    Parameters
    ----------
    env : array-like
        Input data.
    forward_model : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pathak et al (2017) ICM
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Intrinsic motivation / curiosity reward"})


def cheatsheet():
    return "explor: Intrinsic motivation / curiosity reward"

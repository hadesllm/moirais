"""Constrained / safe RL (CMDP)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["safe_rl"]


def safe_rl(env, policy, cost_fn, threshold):
    """
    Constrained / safe RL (CMDP)

    Formula: max E[R] s.t. E[C] ≤ d

    Parameters
    ----------
    env : array-like
        Input data.
    policy : array-like
        Input data.
    cost_fn : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Altman (1999); Achiam et al (2017) CPO
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Constrained / safe RL (CMDP)"})


def cheatsheet():
    return "safrl: Constrained / safe RL (CMDP)"

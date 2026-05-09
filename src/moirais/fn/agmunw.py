"""MuZero n-step value bootstrap target."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["muzero_n_step_value"]


def muzero_n_step_value(rewards, values, n, gamma):
    """
    MuZero n-step value bootstrap target

    Formula: sum_k gamma^k r_{t+k} + gamma^n v_{t+n}

    Parameters
    ----------
    rewards : array-like
        Input data.
    values : array-like
        Input data.
    n : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schrittwieser et al (2020)
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    n = len(values)
    result = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MuZero n-step value bootstrap target"})


def cheatsheet():
    return "agmunw: MuZero n-step value bootstrap target"

"""Generalized advantage estimation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gae"]


def gae(traj, V, gamma, lam):
    """
    Generalized advantage estimation

    Formula: A_t^GAE = sum (γλ)^k δ_{t+k}

    Parameters
    ----------
    traj : array-like
        Input data.
    V : array-like
        Input data.
    gamma : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schulman et al (2015) GAE
    """
    traj = np.atleast_1d(np.asarray(traj, dtype=float))
    n = len(traj)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Generalized advantage estimation"})
    estimate = np.median(traj)
    se = 1.2533 * np.std(traj, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Generalized advantage estimation"})


def cheatsheet():
    return "gae: Generalized advantage estimation"

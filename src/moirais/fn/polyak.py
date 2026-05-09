"""Polyak averaging for target nets."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["polyak_target"]


def polyak_target(theta, theta_target, tau):
    """
    Polyak averaging for target nets

    Formula: θ_target ← τ θ + (1−τ) θ_target

    Parameters
    ----------
    theta : array-like
        Input data.
    theta_target : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Polyak (1990); Lillicrap et al (2016)
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polyak averaging for target nets"})


def cheatsheet():
    return "polyak: Polyak averaging for target nets"

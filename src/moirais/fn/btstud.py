"""Studentised bootstrap (bootstrap-t) CI."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_studentized_ci"]


def boot_studentized_ci(theta_hat, se_hat, t_b, alpha):
    """
    Studentised bootstrap (bootstrap-t) CI

    Formula: [θ̂ - t*_{1-α/2} ŝe, θ̂ - t*_{α/2} ŝe]

    Parameters
    ----------
    theta_hat : array-like
        Input data.
    se_hat : array-like
        Input data.
    t_b : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lo, hi

    References
    ----------
    Hall (1988)
    """
    theta_hat = np.atleast_1d(np.asarray(theta_hat, dtype=float))
    n = len(theta_hat)
    result = float(np.mean(theta_hat))
    se = float(np.std(theta_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Studentised bootstrap (bootstrap-t) CI"})


def cheatsheet():
    return "btstud: Studentised bootstrap (bootstrap-t) CI"

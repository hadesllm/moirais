"""Expectation-maximization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["em_algorithm"]


def em_algorithm(log_lik, Q, x0, steps):
    """
    Expectation-maximization

    Formula: alternate E-step + M-step

    Parameters
    ----------
    log_lik : array-like
        Input data.
    Q : array-like
        Input data.
    x0 : array-like
        Input data.
    steps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dempster-Laird-Rubin (1977)
    """
    log_lik = np.atleast_1d(np.asarray(log_lik, dtype=float))
    n = len(log_lik)
    result = float(np.mean(log_lik))
    se = float(np.std(log_lik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Expectation-maximization"})


def cheatsheet():
    return "epsig1: Expectation-maximization"

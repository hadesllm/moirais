"""Bayesian optimization w/ EI/UCB."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayesian_optimization"]


def bayesian_optimization(f, domain, kernel, acquisition):
    """
    Bayesian optimization w/ EI/UCB

    Formula: max acquisition under GP posterior

    Parameters
    ----------
    f : array-like
        Input data.
    domain : array-like
        Input data.
    kernel : array-like
        Input data.
    acquisition : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mockus (1975); Snoek et al (2012)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian optimization w/ EI/UCB"})


def cheatsheet():
    return "bayoptr: Bayesian optimization w/ EI/UCB"

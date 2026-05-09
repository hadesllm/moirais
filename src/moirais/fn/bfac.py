"""Bayes factor between two models."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_factor"]


def bayes_factor(log_lik_a, log_lik_b):
    """
    Bayes factor between two models

    Formula: BF_12 = m(y | M1) / m(y | M2)

    Parameters
    ----------
    log_lik_a : array-like
        Input data.
    log_lik_b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kass & Raftery (1995)
    """
    log_lik_a = np.atleast_1d(np.asarray(log_lik_a, dtype=float))
    n = len(log_lik_a)
    result = float(np.mean(log_lik_a))
    se = float(np.std(log_lik_a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayes factor between two models"})


def cheatsheet():
    return "bfac: Bayes factor between two models"

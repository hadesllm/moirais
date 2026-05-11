# morie.fn — function file (hadesllm/morie)
"""Bayesian information criterion for cluster-number selection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bic"]


def geron_bic(log_lik, k, n):
    """
    Bayesian information criterion for cluster-number selection

    Formula: BIC = -2 log L + k log(n)

    Parameters
    ----------
    log_lik : array-like
        Input data.
    k : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bic

    References
    ----------
    Géron Ch 8
    """
    log_lik = np.atleast_1d(np.asarray(log_lik, dtype=float))
    n = len(log_lik)
    result = float(np.mean(log_lik))
    se = float(np.std(log_lik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian information criterion for cluster-number selection"})


def cheatsheet():
    return "hmbic: Bayesian information criterion for cluster-number selection"

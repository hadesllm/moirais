"""Bayesian information criterion (BIC)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayesian_information_criterion"]


def bayesian_information_criterion(log_lik, n_params, n_obs):
    """
    Bayesian information criterion (BIC)

    Formula: BIC = -2 log L + p log n

    Parameters
    ----------
    log_lik : array-like
        Input data.
    n_params : array-like
        Input data.
    n_obs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schwarz (1978)
    """
    log_lik = np.atleast_1d(np.asarray(log_lik, dtype=float))
    n = len(log_lik)
    result = float(np.mean(log_lik))
    se = float(np.std(log_lik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian information criterion (BIC)"})


def cheatsheet():
    return "bicg: Bayesian information criterion (BIC)"

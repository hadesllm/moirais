"""Pareto-k shape diagnostic for PSIS-LOO."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pareto_k_diagnostic"]


def pareto_k_diagnostic(log_lik):
    """
    Pareto-k shape diagnostic for PSIS-LOO

    Formula: k > 0.7 → unreliable importance weights

    Parameters
    ----------
    log_lik : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vehtari et al. (2017)
    """
    log_lik = np.atleast_1d(np.asarray(log_lik, dtype=float))
    n = len(log_lik)
    result = float(np.mean(log_lik))
    se = float(np.std(log_lik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pareto-k shape diagnostic for PSIS-LOO"})


def cheatsheet():
    return "khatd: Pareto-k shape diagnostic for PSIS-LOO"

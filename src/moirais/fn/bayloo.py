"""PSIS-LOO leave-one-out."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["loo_psi"]


def loo_psi(log_lik):
    """
    PSIS-LOO leave-one-out

    Formula: Pareto-smoothed importance sampling

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
    Vehtari-Gelman-Gabry (2017)
    """
    log_lik = np.atleast_1d(np.asarray(log_lik, dtype=float))
    n = len(log_lik)
    result = float(np.mean(log_lik))
    se = float(np.std(log_lik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PSIS-LOO leave-one-out"})


def cheatsheet():
    return "bayloo: PSIS-LOO leave-one-out"

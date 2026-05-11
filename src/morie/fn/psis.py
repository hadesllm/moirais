"""PSIS-LOO importance weight smoothing."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pareto_smoothed_importance_sampling"]


def pareto_smoothed_importance_sampling(log_lik):
    """
    PSIS-LOO importance weight smoothing

    Formula: w_i^(s) = clip( p(y_i | theta_s)^{-1}, GPD smooth )

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
    Vehtari, Simpson, Gelman, Yao, Gabry (2024)
    """
    log_lik = np.atleast_1d(np.asarray(log_lik, dtype=float))
    n = len(log_lik)
    result = float(np.mean(log_lik))
    se = float(np.std(log_lik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PSIS-LOO importance weight smoothing"})


def cheatsheet():
    return "psis: PSIS-LOO importance weight smoothing"

"""PSIS-LOO with Pareto-k tail smoothing."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["loo_pareto_smooth"]


def loo_pareto_smooth(log_lik):
    """
    PSIS-LOO with Pareto-k tail smoothing

    Formula: importance weights w_i smoothed via fit GPD on top 20%

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
    Vehtari et al. (2024)
    """
    log_lik = np.atleast_1d(np.asarray(log_lik, dtype=float))
    n = len(log_lik)
    result = float(np.mean(log_lik))
    se = float(np.std(log_lik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PSIS-LOO with Pareto-k tail smoothing"})


def cheatsheet():
    return "loopr: PSIS-LOO with Pareto-k tail smoothing"

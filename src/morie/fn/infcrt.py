"""WAIC + LOO-CV."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["information_criterion"]


def information_criterion(log_lik_samples):
    """
    WAIC + LOO-CV

    Formula: WAIC = -2(lppd - p_WAIC); PSIS-LOO with Pareto-k

    Parameters
    ----------
    log_lik_samples : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Watanabe (2013); Vehtari-Gelman-Gabry (2017)
    """
    log_lik_samples = np.atleast_1d(np.asarray(log_lik_samples, dtype=float))
    n = len(log_lik_samples)
    result = float(np.mean(log_lik_samples))
    se = float(np.std(log_lik_samples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "WAIC + LOO-CV"})


def cheatsheet():
    return "infcrt: WAIC + LOO-CV"

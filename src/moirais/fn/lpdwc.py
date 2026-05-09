"""Log pointwise predictive density (lppd)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["log_pointwise_predictive_density"]


def log_pointwise_predictive_density(log_lik):
    """
    Log pointwise predictive density (lppd)

    Formula: lppd = sum_i log( (1/S) sum_s p(y_i | theta_s) )

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
    Gelman et al. BDA3 §7.2
    """
    log_lik = np.atleast_1d(np.asarray(log_lik, dtype=float))
    n = len(log_lik)
    result = float(np.mean(log_lik))
    se = float(np.std(log_lik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log pointwise predictive density (lppd)"})


def cheatsheet():
    return "lpdwc: Log pointwise predictive density (lppd)"

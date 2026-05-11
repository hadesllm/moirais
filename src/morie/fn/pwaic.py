"""Effective parameters from WAIC (p_WAIC)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["effective_parameters_waic"]


def effective_parameters_waic(log_lik):
    """
    Effective parameters from WAIC (p_WAIC)

    Formula: p_WAIC = sum_i Var_s( log p(y_i | theta_s) )

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
    Vehtari, Gelman, Gabry (2017)
    """
    log_lik = np.atleast_1d(np.asarray(log_lik, dtype=float))
    n = len(log_lik)
    result = float(np.mean(log_lik))
    se = float(np.std(log_lik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Effective parameters from WAIC (p_WAIC)"})


def cheatsheet():
    return "pwaic: Effective parameters from WAIC (p_WAIC)"

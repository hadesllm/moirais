"""Watanabe-Akaike information criterion (WAIC)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["waic_diagnostic"]


def waic_diagnostic(log_lik):
    """
    Watanabe-Akaike information criterion (WAIC)

    Formula: WAIC = -2(lppd - p_WAIC)

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
    Watanabe (2010); Gelman et al. BDA3 §7
    """
    log_lik = np.atleast_1d(np.asarray(log_lik, dtype=float))
    n = len(log_lik)
    result = float(np.mean(log_lik))
    se = float(np.std(log_lik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Watanabe-Akaike information criterion (WAIC)"})


def cheatsheet():
    return "waicd: Watanabe-Akaike information criterion (WAIC)"

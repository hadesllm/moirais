# moirais.fn — function file (hadesllm/moirais)
"""Akaike information criterion for cluster-number selection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_aic"]


def geron_aic(log_lik, k):
    """
    Akaike information criterion for cluster-number selection

    Formula: AIC = -2 log L + 2k

    Parameters
    ----------
    log_lik : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: aic

    References
    ----------
    Géron Ch 8
    """
    log_lik = np.atleast_1d(np.asarray(log_lik, dtype=float))
    n = len(log_lik)
    result = float(np.mean(log_lik))
    se = float(np.std(log_lik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Akaike information criterion for cluster-number selection"})


def cheatsheet():
    return "hmaic: Akaike information criterion for cluster-number selection"

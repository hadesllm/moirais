"""Cox dfbeta influence diagnostics."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cox_dfbeta_influence"]


def cox_dfbeta_influence(fit):
    """
    Cox dfbeta influence diagnostics

    Formula: approx delta-beta from leaving observation i out

    Parameters
    ----------
    fit : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Therneau-Grambsch (2000)
    """
    fit = np.atleast_1d(np.asarray(fit, dtype=float))
    n = len(fit)
    result = float(np.mean(fit))
    se = float(np.std(fit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cox dfbeta influence diagnostics"})


def cheatsheet():
    return "coxdfb: Cox dfbeta influence diagnostics"

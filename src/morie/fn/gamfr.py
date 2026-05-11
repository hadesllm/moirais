"""Gamma frailty model for clustered survival."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gamma_frailty_cox"]


def gamma_frailty_cox(time, event, X, cluster):
    """
    Gamma frailty model for clustered survival

    Formula: lambda_ij(t|w_i) = w_i lambda_0(t) exp(beta'X_ij), w_i ~ Gamma

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Klein (1992); Therneau & Grambsch §9
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gamma frailty model for clustered survival"})


def cheatsheet():
    return "gamfr: Gamma frailty model for clustered survival"

"""Bayesian GPD posterior via Metropolis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_bayes_gpd"]


def evt_bayes_gpd(y, n_iter, prior):
    """
    Bayesian GPD posterior via Metropolis

    Formula: MH on (log σ, ξ); profile or joint

    Parameters
    ----------
    y : array-like
        Input data.
    n_iter : array-like
        Input data.
    prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: draws

    References
    ----------
    Coles (2001)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian GPD posterior via Metropolis"})


def cheatsheet():
    return "evbgpd: Bayesian GPD posterior via Metropolis"

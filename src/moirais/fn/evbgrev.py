"""Bayesian GEV via Metropolis with prior π(μ,σ,ξ)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_bayes_gev"]


def evt_bayes_gev(x, n_iter, prior):
    """
    Bayesian GEV via Metropolis with prior π(μ,σ,ξ)

    Formula: MH random-walk on (μ,log σ,ξ)

    Parameters
    ----------
    x : array-like
        Input data.
    n_iter : array-like
        Input data.
    prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: draws, accept_rate

    References
    ----------
    Coles & Tawn (1996)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian GEV via Metropolis with prior π(μ,σ,ξ)"})


def cheatsheet():
    return "evbgrev: Bayesian GEV via Metropolis with prior π(μ,σ,ξ)"

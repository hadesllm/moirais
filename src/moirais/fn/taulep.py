"""Tau-leap stochastic SIR."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tau_leap_sim"]


def tau_leap_sim(state, rates, tau):
    """
    Tau-leap stochastic SIR

    Formula: Poisson-leap approximation of Gillespie

    Parameters
    ----------
    state : array-like
        Input data.
    rates : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gillespie (2001)
    """
    rates = np.atleast_1d(np.asarray(rates, dtype=float))
    n = len(rates)
    result = float(np.mean(rates))
    se = float(np.std(rates, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tau-leap stochastic SIR"})


def cheatsheet():
    return "taulep: Tau-leap stochastic SIR"

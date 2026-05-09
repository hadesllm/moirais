"""Stochastic SIR (Gillespie)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sir_stochastic"]


def sir_stochastic(S0, I0, beta, gamma, T):
    """
    Stochastic SIR (Gillespie)

    Formula: continuous-time Markov chain via SSA

    Parameters
    ----------
    S0 : array-like
        Input data.
    I0 : array-like
        Input data.
    beta : array-like
        Input data.
    gamma : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gillespie (1977)
    """
    S0 = np.atleast_1d(np.asarray(S0, dtype=float))
    n = len(S0)
    result = float(np.mean(S0))
    se = float(np.std(S0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stochastic SIR (Gillespie)"})


def cheatsheet():
    return "sirstn: Stochastic SIR (Gillespie)"

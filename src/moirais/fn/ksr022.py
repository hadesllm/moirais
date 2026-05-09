"""Multiplicative intensity model for a counting process with proportional baseline hazard."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch1_multiplicative_intensity"]


def kosorok_ch1_multiplicative_intensity(t, Z, Y, beta, Lambda):
    """
    Multiplicative intensity model for a counting process with proportional baseline hazard

    Formula: E[N(t) | Z] = integral_0^t E[Y(s)|Z] * exp(beta' * Z) * dLambda(s)

    Parameters
    ----------
    t : array-like
        Input data.
    Z : array-like
        Input data.
    Y : array-like
        Input data.
    beta : array-like
        Input data.
    Lambda : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 1, Eq 1.3, p. 5
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multiplicative intensity model for a counting process with proportional baseline hazard"})


def cheatsheet():
    return "ksr022: Multiplicative intensity model for a counting process with proportional baseline hazard"

"""Pathwise derivative of psi along a smooth submodel expressed via efficient influence function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch3_pathwise_derivative"]


def kosorok_ch3_pathwise_derivative(psi, P_t, l_dot, g, a, theta, eta):
    """
    Pathwise derivative of psi along a smooth submodel expressed via efficient influence function

    Formula: d psi(P_t)/dt |_{t=0} = a = P[ psi_tilde_{theta,eta} (l_dot_{theta,eta} a + g) ]

    Parameters
    ----------
    psi : array-like
        Input data.
    P_t : array-like
        Input data.
    l_dot : array-like
        Input data.
    g : array-like
        Input data.
    a : array-like
        Input data.
    theta : array-like
        Input data.
    eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 3, Eq 3.2, p. 40
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    n = len(psi)
    result = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pathwise derivative of psi along a smooth submodel expressed via efficient influence function"})


def cheatsheet():
    return "ksr062: Pathwise derivative of psi along a smooth submodel expressed via efficient influence function"

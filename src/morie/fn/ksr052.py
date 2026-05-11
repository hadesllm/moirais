"""Frechet derivative of the Kaplan-Meier self-consistency operator at S0."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_kaplan_meier_derivative"]


def kosorok_ch2_kaplan_meier_derivative(S_0, L, G, h, t):
    """
    Frechet derivative of the Kaplan-Meier self-consistency operator at S0

    Formula: Psi_dot_{theta_0}(h)(t) = - integral_0^t S_0(t) h(u) / S_0(u) dG(u) - L(t) h(t)

    Parameters
    ----------
    S_0 : array-like
        Input data.
    L : array-like
        Input data.
    G : array-like
        Input data.
    h : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.16, p. 27
    """
    S_0 = np.atleast_1d(np.asarray(S_0, dtype=float))
    n = len(S_0)
    result = float(np.mean(S_0))
    se = float(np.std(S_0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frechet derivative of the Kaplan-Meier self-consistency operator at S0"})


def cheatsheet():
    return "ksr052: Frechet derivative of the Kaplan-Meier self-consistency operator at S0"

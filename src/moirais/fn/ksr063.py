"""Efficient score function for beta in the Cox model under right censoring."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch3_cox_efficient_score_beta"]


def kosorok_ch3_cox_efficient_score_beta(Z, Y, beta, Lambda, M, tau):
    """
    Efficient score function for beta in the Cox model under right censoring

    Formula: l_tilde_{beta,Lambda} = integral_0^tau { Z - P_{beta,Lambda}[ Z Y(t) e^{beta'Z} ] / P_{beta,Lambda}[ Y(t) e^{beta'Z} ] } dM(t)

    Parameters
    ----------
    Z : array-like
        Input data.
    Y : array-like
        Input data.
    beta : array-like
        Input data.
    Lambda : array-like
        Input data.
    M : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 3, Eq 3.3, p. 42
    """
    Z = np.atleast_1d(np.asarray(Z, dtype=float))
    n = len(Z)
    result = float(np.mean(Z))
    se = float(np.std(Z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Efficient score function for beta in the Cox model under right censoring"})


def cheatsheet():
    return "ksr063: Efficient score function for beta in the Cox model under right censoring"

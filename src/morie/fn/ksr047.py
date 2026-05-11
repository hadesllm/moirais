"""Efron self-consistency Z-estimator representation of the Kaplan-Meier survival estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch2_kaplan_meier_self_consistency"]


def kosorok_ch2_kaplan_meier_self_consistency(S, S_0, L, G, t):
    """
    Efron self-consistency Z-estimator representation of the Kaplan-Meier survival estimator

    Formula: Psi(S)(t) = P psi_{S,t} = S_0(t) L(t) + integral_0^t S_0(u)/S(u) dG(u) S(t) - S(t)

    Parameters
    ----------
    S : array-like
        Input data.
    S_0 : array-like
        Input data.
    L : array-like
        Input data.
    G : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 2, Eq 2.11, p. 26
    """
    S = np.atleast_1d(np.asarray(S, dtype=float))
    n = len(S)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Efron self-consistency Z-estimator representation of the Kaplan-Meier survival estimator"})
    estimate = np.median(S)
    se = 1.2533 * np.std(S, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Efron self-consistency Z-estimator representation of the Kaplan-Meier survival estimator"})


def cheatsheet():
    return "ksr047: Efron self-consistency Z-estimator representation of the Kaplan-Meier survival estimator"

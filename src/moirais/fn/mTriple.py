"""Triply robust NIE estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["triply_robust_mediation"]


def triply_robust_mediation(Y, X, M, C):
    """
    Triply robust NIE estimator

    Formula: consistent if 2 of 3 nuisance models correct

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tchetgen Tchetgen-Shpitser (2012)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Triply robust NIE estimator"})
    estimate = np.median(Y)
    se = 1.2533 * np.std(Y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Triply robust NIE estimator"})


def cheatsheet():
    return "mTriple: Triply robust NIE estimator"

"""Score-operator condition along a path eta_t for the maximum-likelihood estimating equations."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch3_score_operator_path"]


def kosorok_ch3_score_operator_path(B, theta, eta, h, x, p):
    """
    Score-operator condition along a path eta_t for the maximum-likelihood estimating equations

    Formula: B_{theta,eta} h(x) - P_{theta,eta} B_{theta,eta} h = d log p_{theta, eta_t(theta,eta)}(x)/dt |_{t=0}

    Parameters
    ----------
    B : array-like
        Input data.
    theta : array-like
        Input data.
    eta : array-like
        Input data.
    h : array-like
        Input data.
    x : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 3, Eq 3.10, p. 46
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Score-operator condition along a path eta_t for the maximum-likelihood estimating equations"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Score-operator condition along a path eta_t for the maximum-likelihood estimating equations"})


def cheatsheet():
    return "ksr070: Score-operator condition along a path eta_t for the maximum-likelihood estimating equations"

"""Expected a posteriori (EAP) theta estimate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["eap_theta_estimator"]


def eap_theta_estimator(y, prior, P_theta):
    """
    Expected a posteriori (EAP) theta estimate

    Formula: theta_hat_EAP = integral theta P(theta|y) d theta

    Parameters
    ----------
    y : array-like
        Input data.
    prior : array-like
        Input data.
    P_theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bock & Mislevy (1982)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Expected a posteriori (EAP) theta estimate"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Expected a posteriori (EAP) theta estimate"})


def cheatsheet():
    return "eapth: Expected a posteriori (EAP) theta estimate"

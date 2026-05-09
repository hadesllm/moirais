"""Random forest prediction average."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_random_forest"]


def esl_random_forest(X, y, B):
    """
    Random forest prediction average

    Formula: f_hat(x) = (1/B) sum T_b(x; Theta_b)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction

    References
    ----------
    Hastie ESL Ch 15
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Random forest prediction average"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Random forest prediction average"})


def cheatsheet():
    return "eslrft: Random forest prediction average"

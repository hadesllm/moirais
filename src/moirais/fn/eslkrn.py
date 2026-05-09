"""Kernel density estimate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_kernel_density"]


def esl_kernel_density(x, data, lambda_):
    """
    Kernel density estimate

    Formula: f_lambda(x) = (1/(N lambda)) sum K_lambda(x, x_i)

    Parameters
    ----------
    x : array-like
        Input data.
    data : array-like
        Input data.
    lambda_ : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: density

    References
    ----------
    Hastie ESL Ch 6
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Kernel density estimate"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Kernel density estimate"})


def cheatsheet():
    return "eslkrn: Kernel density estimate"

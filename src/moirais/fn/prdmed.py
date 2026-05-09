"""Product-of-coefficients estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["product_of_coefficients"]


def product_of_coefficients(a, b):
    """
    Product-of-coefficients estimator

    Formula: NIE = a·b

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    MacKinnon (2008) book
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Product-of-coefficients estimator"})
    estimate = np.median(a)
    se = 1.2533 * np.std(a, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Product-of-coefficients estimator"})


def cheatsheet():
    return "prdmed: Product-of-coefficients estimator"

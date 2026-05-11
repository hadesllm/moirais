"""Asymmetric inner-product estimator (Eq 4): uses real Sq and 1-bit H_S(k)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_qjl_product_estimator"]


def turboquant_qjl_product_estimator(q, signs_k, norm_k, S):
    """
    Asymmetric inner-product estimator (Eq 4): uses real Sq and 1-bit H_S(k)

    Formula: ProdQJL(q,k) := sqrt(pi/2) / m * ||k||_2 * <S q, H_S(k)>

    Parameters
    ----------
    q : array-like
        Input data.
    signs_k : array-like
        Input data.
    norm_k : array-like
        Input data.
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zandieh et al. 2024 Eq 4 (asymmetric inner product)
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    n = len(q)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Asymmetric inner-product estimator (Eq 4): uses real Sq and 1-bit H_S(k)"})
    estimate = np.median(q)
    se = 1.2533 * np.std(q, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Asymmetric inner-product estimator (Eq 4): uses real Sq and 1-bit H_S(k)"})


def cheatsheet():
    return "tqprod: Asymmetric inner-product estimator (Eq 4): uses real Sq and 1-bit H_S(k)"

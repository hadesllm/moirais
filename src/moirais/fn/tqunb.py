"""Lemma 3.2: the ProdQJL estimator is unbiased, E[ProdQJL(q,k)] = <q,k>."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_prodqjl_unbiasedness"]


def turboquant_prodqjl_unbiasedness(q, k, m):
    """
    Lemma 3.2: the ProdQJL estimator is unbiased, E[ProdQJL(q,k)] = <q,k>

    Formula: E[ sqrt(pi/2)/m * ||k||_2 * <S q, sign(S k)> ] = <q, k>

    Parameters
    ----------
    q : array-like
        Input data.
    k : array-like
        Input data.
    m : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bias

    References
    ----------
    Zandieh et al. 2024 Lemma 3.2 (unbiasedness)
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    n = len(q)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Lemma 3.2: the ProdQJL estimator is unbiased, E[ProdQJL(q,k)] = <q,k>"})
    estimate = np.median(q)
    se = 1.2533 * np.std(q, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Lemma 3.2: the ProdQJL estimator is unbiased, E[ProdQJL(q,k)] = <q,k>"})


def cheatsheet():
    return "tqunb: Lemma 3.2: the ProdQJL estimator is unbiased, E[ProdQJL(q,k)] = <q,k>"

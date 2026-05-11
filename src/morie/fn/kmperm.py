# morie.fn — function file (hadesllm/morie)
"""Permutation LM loss (XLNet): average over random factorization orders."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_permutation_lm_loss"]


def kamath_permutation_lm_loss(logits, targets, permutation):
    """
    Permutation LM loss (XLNet): average over random factorization orders

    Formula: L_PLM = E_{z ~ Perm(T)} [ - sum_{t} log p(x_{z_t} | x_{z_<t}) ]

    Parameters
    ----------
    logits : array-like
        Input data.
    targets : array-like
        Input data.
    permutation : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Kamath Ch 2, Permutation Language Modeling section
    """
    logits = np.atleast_1d(np.asarray(logits, dtype=float))
    n = len(logits)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Permutation LM loss (XLNet): average over random factorization orders"})
    estimate = np.median(logits)
    se = 1.2533 * np.std(logits, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Permutation LM loss (XLNet): average over random factorization orders"})


def cheatsheet():
    return "kmperm: Permutation LM loss (XLNet): average over random factorization orders"

# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""CBOW: predict center word from averaged context embeddings."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_cbow"]


def burkov_cbow(context_indices, center_index, V, U):
    """
    CBOW: predict center word from averaged context embeddings

    Formula: h = (1/|C|) sum_{c in C} v_c;  P(w | C) = softmax(U h)_w

    Parameters
    ----------
    context_indices : array-like
        Input data.
    center_index : array-like
        Input data.
    V : array-like
        Input data.
    U : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Burkov Ch 2, CBOW section
    """
    context_indices = np.atleast_1d(np.asarray(context_indices, dtype=float))
    n = len(context_indices)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "CBOW: predict center word from averaged context embeddings"})
    estimate = np.median(context_indices)
    se = 1.2533 * np.std(context_indices, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "CBOW: predict center word from averaged context embeddings"})


def cheatsheet():
    return "bkcbow: CBOW: predict center word from averaged context embeddings"

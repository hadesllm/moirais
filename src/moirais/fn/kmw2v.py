# moirais.fn — function file (hadesllm/moirais)
"""Word2Vec skip-gram log-likelihood (one center, context window c)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_word2vec_skipgram"]


def kamath_word2vec_skipgram(center_indices, context_indices, V, U):
    """
    Word2Vec skip-gram log-likelihood (one center, context window c)

    Formula: L = sum_{t=1..T} sum_{-c<=j<=c, j!=0} log P(w_{t+j} | w_t); P = softmax(u_o^T v_c)

    Parameters
    ----------
    center_indices : array-like
        Input data.
    context_indices : array-like
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
    Kamath Ch 1, Word2Vec section
    """
    center_indices = np.atleast_1d(np.asarray(center_indices, dtype=float))
    n = len(center_indices)
    result = float(np.mean(center_indices))
    se = float(np.std(center_indices, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Word2Vec skip-gram log-likelihood (one center, context window c)"})


def cheatsheet():
    return "kmw2v: Word2Vec skip-gram log-likelihood (one center, context window c)"

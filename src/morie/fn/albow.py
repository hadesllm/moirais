# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bag-of-words vector: per-document term-count vector over the fixed vocabulary."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_bag_of_words"]


def alammar_bag_of_words(tokens, vocab):
    """
    Bag-of-words vector: per-document term-count vector over the fixed vocabulary

    Formula: bow_d[v] = count(v in d)

    Parameters
    ----------
    tokens : array-like
        Input data.
    vocab : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bow_vector

    References
    ----------
    Alammar Ch 1, Bag-of-Words section
    """
    tokens = np.atleast_1d(np.asarray(tokens, dtype=float))
    n = len(tokens)
    result = float(np.mean(tokens))
    se = float(np.std(tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bag-of-words vector: per-document term-count vector over the fixed vocabulary"})


def cheatsheet():
    return "albow: Bag-of-words vector: per-document term-count vector over the fixed vocabulary"

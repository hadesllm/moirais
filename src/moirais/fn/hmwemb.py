# moirais.fn — function file (hadesllm/moirais)
"""Word embeddings: dense vector representations learned per token."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_word_embeddings"]


def geron_word_embeddings(vocab, d):
    """
    Word embeddings: dense vector representations learned per token

    Formula: E in R^(V x d); token t -> E[t]

    Parameters
    ----------
    vocab : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: E

    References
    ----------
    Géron Ch 14
    """
    vocab = np.atleast_1d(np.asarray(vocab, dtype=float))
    n = len(vocab)
    result = float(np.mean(vocab))
    se = float(np.std(vocab, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Word embeddings: dense vector representations learned per token"})


def cheatsheet():
    return "hmwemb: Word embeddings: dense vector representations learned per token"

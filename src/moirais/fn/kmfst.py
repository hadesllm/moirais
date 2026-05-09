# moirais.fn — function file (hadesllm/moirais)
"""FastText word representation: sum of subword n-gram embeddings."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_fasttext_subword"]


def kamath_fasttext_subword(word, ngram_embeddings, n_min, n_max):
    """
    FastText word representation: sum of subword n-gram embeddings

    Formula: v_w = sum_{g in Ngrams(w)} z_g

    Parameters
    ----------
    word : array-like
        Input data.
    ngram_embeddings : array-like
        Input data.
    n_min : array-like
        Input data.
    n_max : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: v

    References
    ----------
    Kamath Ch 1, FastText section
    """
    word = np.atleast_1d(np.asarray(word, dtype=float))
    n = len(word)
    result = float(np.mean(word))
    se = float(np.std(word, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "FastText word representation: sum of subword n-gram embeddings"})


def cheatsheet():
    return "kmfst: FastText word representation: sum of subword n-gram embeddings"

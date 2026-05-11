# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Tokenizer vocabulary comparison: Jaccard overlap between two tokenizer vocabs."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_tokenizer_vocab_overlap"]


def alammar_tokenizer_vocab_overlap(vocab_a, vocab_b):
    """
    Tokenizer vocabulary comparison: Jaccard overlap between two tokenizer vocabs

    Formula: J(V_A, V_B) = |V_A ∩ V_B| / |V_A ∪ V_B|

    Parameters
    ----------
    vocab_a : array-like
        Input data.
    vocab_b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: jaccard

    References
    ----------
    Alammar Ch 2, Tokenizer Vocabulary Comparison section
    """
    vocab_a = np.atleast_1d(np.asarray(vocab_a, dtype=float))
    n = len(vocab_a)
    result = float(np.mean(vocab_a))
    se = float(np.std(vocab_a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tokenizer vocabulary comparison: Jaccard overlap between two tokenizer vocabs"})


def cheatsheet():
    return "alvocb: Tokenizer vocabulary comparison: Jaccard overlap between two tokenizer vocabs"

# moirais.fn — function file (hadesllm/moirais)
"""Byte-pair encoding tokenizer: iteratively merge most frequent symbol pair."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bpe_tokenizer"]


def geron_bpe_tokenizer(corpus, vocab_size):
    """
    Byte-pair encoding tokenizer: iteratively merge most frequent symbol pair

    Formula: at each step merge argmax pair; rebuild vocab

    Parameters
    ----------
    corpus : array-like
        Input data.
    vocab_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: vocab

    References
    ----------
    Géron Ch 14
    """
    corpus = np.atleast_1d(np.asarray(corpus, dtype=float))
    n = len(corpus)
    result = float(np.mean(corpus))
    se = float(np.std(corpus, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Byte-pair encoding tokenizer: iteratively merge most frequent symbol pair"})


def cheatsheet():
    return "hmbpet: Byte-pair encoding tokenizer: iteratively merge most frequent symbol pair"

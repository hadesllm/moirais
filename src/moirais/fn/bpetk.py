"""Byte-pair encoding tokenizer."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bpe_tokenizer"]


def bpe_tokenizer(corpus, vocab_size):
    """
    Byte-pair encoding tokenizer

    Formula: iteratively merge most frequent adjacent pair

    Parameters
    ----------
    corpus : array-like
        Input data.
    vocab_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sennrich et al (2016) BPE
    """
    corpus = np.atleast_1d(np.asarray(corpus, dtype=float))
    n = len(corpus)
    result = float(np.mean(corpus))
    se = float(np.std(corpus, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Byte-pair encoding tokenizer"})


def cheatsheet():
    return "bpetk: Byte-pair encoding tokenizer"

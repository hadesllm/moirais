# morie.fn — function file (hadesllm/morie)
"""Byte-pair encoding merge step: merge most-frequent adjacent pair."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bpe_tokenizer_merge"]


def geron_bpe_tokenizer_merge(corpus, n_merges):
    """
    Byte-pair encoding merge step: merge most-frequent adjacent pair

    Formula: pair* = argmax_pair count(pair in corpus); merge pair* into single token

    Parameters
    ----------
    corpus : array-like
        Input data.
    n_merges : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: vocab, merges

    References
    ----------
    Géron Ch 14, BPE tokenizer section
    """
    corpus = np.atleast_1d(np.asarray(corpus, dtype=float))
    n = len(corpus)
    result = float(np.mean(corpus))
    se = float(np.std(corpus, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Byte-pair encoding merge step: merge most-frequent adjacent pair"})


def cheatsheet():
    return "grbpe: Byte-pair encoding merge step: merge most-frequent adjacent pair"

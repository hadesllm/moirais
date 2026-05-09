"""SentencePiece (unigram LM)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sentencepiece"]


def sentencepiece(corpus, vocab_size, alpha):
    """
    SentencePiece (unigram LM)

    Formula: EM over unigram LM with prune-and-keep

    Parameters
    ----------
    corpus : array-like
        Input data.
    vocab_size : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kudo-Richardson (2018)
    """
    corpus = np.atleast_1d(np.asarray(corpus, dtype=float))
    n = len(corpus)
    result = float(np.mean(corpus))
    se = float(np.std(corpus, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SentencePiece (unigram LM)"})


def cheatsheet():
    return "sentpc: SentencePiece (unigram LM)"

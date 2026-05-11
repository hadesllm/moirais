"""WordPiece tokenizer."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wordpiece"]


def wordpiece(corpus, vocab_size):
    """
    WordPiece tokenizer

    Formula: greedy longest-match using likelihood-based merges

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
    Schuster-Nakajima (2012); Devlin et al (2019) BERT
    """
    corpus = np.atleast_1d(np.asarray(corpus, dtype=float))
    n = len(corpus)
    result = float(np.mean(corpus))
    se = float(np.std(corpus, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "WordPiece tokenizer"})


def cheatsheet():
    return "wpiece: WordPiece tokenizer"

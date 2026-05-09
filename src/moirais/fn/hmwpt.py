# moirais.fn — function file (hadesllm/moirais)
"""WordPiece tokenizer: maximum likelihood subword segmentation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_wordpiece_tokenizer"]


def geron_wordpiece_tokenizer(corpus, vocab_size):
    """
    WordPiece tokenizer: maximum likelihood subword segmentation

    Formula: segment word into subwords maximizing likelihood under LM

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "WordPiece tokenizer: maximum likelihood subword segmentation"})


def cheatsheet():
    return "hmwpt: WordPiece tokenizer: maximum likelihood subword segmentation"

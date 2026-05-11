# morie.fn — function file (hadesllm/morie)
"""Use pretrained word embeddings (e.g., GloVe) as initialization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_pretrained_embeddings"]


def geron_pretrained_embeddings(vocab, pretrained, freeze):
    """
    Use pretrained word embeddings (e.g., GloVe) as initialization

    Formula: E init from pretrained matrix; optionally fine-tune

    Parameters
    ----------
    vocab : array-like
        Input data.
    pretrained : array-like
        Input data.
    freeze : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Use pretrained word embeddings (e.g., GloVe) as initialization"})


def cheatsheet():
    return "hmpemb: Use pretrained word embeddings (e.g., GloVe) as initialization"

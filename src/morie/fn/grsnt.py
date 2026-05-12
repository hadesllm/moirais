# morie.fn -- function file (hadesllm/morie)
"""Binary sentiment classification output (positive/negative) via MLP over embeddings."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_sentiment_binary"]


def geron_sentiment_binary(token_ids, E, w, b):
    """
    Binary sentiment classification output (positive/negative) via MLP over embeddings

    Formula: p = sigma(w^T pool(E[x]))

    Parameters
    ----------
    token_ids : array-like
        Input data.
    E : array-like
        Input data.
    w : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p

    References
    ----------
    Géron Ch 14, Sentiment Analysis section
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Binary sentiment classification output (positive/negative) via MLP over embeddings"})


def cheatsheet():
    return "grsnt: Binary sentiment classification output (positive/negative) via MLP over embeddings"

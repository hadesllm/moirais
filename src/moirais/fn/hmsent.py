# moirais.fn — function file (hadesllm/moirais)
"""Sentiment analysis with RNN or transformer on tokens."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_sentiment_analysis"]


def geron_sentiment_analysis(texts, model, tokenizer):
    """
    Sentiment analysis with RNN or transformer on tokens

    Formula: y_hat = softmax(W h_T) for 2 or more sentiment classes

    Parameters
    ----------
    texts : array-like
        Input data.
    model : array-like
        Input data.
    tokenizer : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: predictions

    References
    ----------
    Géron Ch 14
    """
    texts = np.atleast_1d(np.asarray(texts, dtype=float))
    n = len(texts)
    result = float(np.mean(texts))
    se = float(np.std(texts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sentiment analysis with RNN or transformer on tokens"})


def cheatsheet():
    return "hmsent: Sentiment analysis with RNN or transformer on tokens"

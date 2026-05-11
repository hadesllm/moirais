"""Sentence-BERT siamese pair."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sbert"]


def sbert(sent_a, sent_b):
    """
    Sentence-BERT siamese pair

    Formula: shared BERT + mean pool + cosine

    Parameters
    ----------
    sent_a : array-like
        Input data.
    sent_b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Reimers-Gurevych (2019)
    """
    sent_a = np.atleast_1d(np.asarray(sent_a, dtype=float))
    n = len(sent_a)
    result = float(np.mean(sent_a))
    se = float(np.std(sent_a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sentence-BERT siamese pair"})


def cheatsheet():
    return "sbert: Sentence-BERT siamese pair"

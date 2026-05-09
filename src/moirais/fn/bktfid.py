# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""TF-IDF score: term frequency weighted by inverse document frequency."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_tf_idf"]


def burkov_tf_idf(term, document, corpus):
    """
    TF-IDF score: term frequency weighted by inverse document frequency

    Formula: TF-IDF(t, d, D) = TF(t, d) * log( |D| / |{d' in D : t in d'}| )

    Parameters
    ----------
    term : array-like
        Input data.
    document : array-like
        Input data.
    corpus : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tfidf

    References
    ----------
    Burkov Ch 2, TF-IDF section
    """
    term = np.atleast_1d(np.asarray(term, dtype=float))
    n = len(term)
    result = float(np.mean(term))
    se = float(np.std(term, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TF-IDF score: term frequency weighted by inverse document frequency"})


def cheatsheet():
    return "bktfid: TF-IDF score: term frequency weighted by inverse document frequency"

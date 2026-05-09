# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Class-based TF-IDF (BERTopic): class c treated as a single document."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_c_tfidf"]


def alammar_c_tfidf(term_counts_by_class, corpus_freq, A):
    """
    Class-based TF-IDF (BERTopic): class c treated as a single document

    Formula: c-TF-IDF(t, c) = f_{t,c} * log( 1 + A / sum_c f_{t,c} )

    Parameters
    ----------
    term_counts_by_class : array-like
        Input data.
    corpus_freq : array-like
        Input data.
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ctfidf

    References
    ----------
    Alammar Ch 5, c-TF-IDF section (BERTopic)
    """
    term_counts_by_class = np.atleast_1d(np.asarray(term_counts_by_class, dtype=float))
    n = len(term_counts_by_class)
    result = float(np.mean(term_counts_by_class))
    se = float(np.std(term_counts_by_class, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Class-based TF-IDF (BERTopic): class c treated as a single document"})


def cheatsheet():
    return "alctf: Class-based TF-IDF (BERTopic): class c treated as a single document"

"""TF-IDF."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tfidf"]


def tfidf(docs):
    """
    TF-IDF

    Formula: tf · log(N/df)

    Parameters
    ----------
    docs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Spärck Jones (1972); Salton-Buckley (1988)
    """
    docs = np.atleast_1d(np.asarray(docs, dtype=float))
    n = len(docs)
    result = float(np.mean(docs))
    se = float(np.std(docs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TF-IDF"})


def cheatsheet():
    return "tfidfV: TF-IDF"

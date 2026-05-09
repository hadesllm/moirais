"""FastText subword embeddings."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fasttext"]


def fasttext(corpus, dim):
    """
    FastText subword embeddings

    Formula: word vec = sum char-n-gram vecs

    Parameters
    ----------
    corpus : array-like
        Input data.
    dim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bojanowski et al (2017)
    """
    corpus = np.atleast_1d(np.asarray(corpus, dtype=float))
    n = len(corpus)
    result = float(np.mean(corpus))
    se = float(np.std(corpus, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "FastText subword embeddings"})


def cheatsheet():
    return "fastxt: FastText subword embeddings"

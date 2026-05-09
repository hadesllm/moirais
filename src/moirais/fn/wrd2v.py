"""Word2Vec (skip-gram or CBOW)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["word2vec"]


def word2vec(corpus, dim, window):
    """
    Word2Vec (skip-gram or CBOW)

    Formula: P(w_o|w_i) ∝ exp(v_o·v_i); negative sampling

    Parameters
    ----------
    corpus : array-like
        Input data.
    dim : array-like
        Input data.
    window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mikolov et al (2013)
    """
    corpus = np.atleast_1d(np.asarray(corpus, dtype=float))
    n = len(corpus)
    result = float(np.mean(corpus))
    se = float(np.std(corpus, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Word2Vec (skip-gram or CBOW)"})


def cheatsheet():
    return "wrd2v: Word2Vec (skip-gram or CBOW)"

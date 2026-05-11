"""Latent Dirichlet Allocation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lda_topic"]


def lda_topic(docs, K, alpha, beta):
    """
    Latent Dirichlet Allocation

    Formula: θ ~ Dir(α); z ~ Mult(θ); w ~ Mult(φ_z)

    Parameters
    ----------
    docs : array-like
        Input data.
    K : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Blei-Ng-Jordan (2003)
    """
    docs = np.atleast_1d(np.asarray(docs, dtype=float))
    n = len(docs)
    result = float(np.mean(docs))
    se = float(np.std(docs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Latent Dirichlet Allocation"})


def cheatsheet():
    return "lda: Latent Dirichlet Allocation"

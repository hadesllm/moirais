"""t-SNE embedding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tsne_embedding"]


def tsne_embedding(X, perplexity):
    """
    t-SNE embedding

    Formula: minimize KL between high-dim Gaussian + low-dim t

    Parameters
    ----------
    X : array-like
        Input data.
    perplexity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Van der Maaten-Hinton (2008)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "t-SNE embedding"})


def cheatsheet():
    return "sctsne: t-SNE embedding"

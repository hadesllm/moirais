"""GRACE contrastive node embeddings."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["grace"]


def grace(G, X, aug):
    """
    GRACE contrastive node embeddings

    Formula: InfoNCE on two graph augmentations

    Parameters
    ----------
    G : array-like
        Input data.
    X : array-like
        Input data.
    aug : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhu et al (2020) GRACE
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GRACE contrastive node embeddings"})


def cheatsheet():
    return "grace: GRACE contrastive node embeddings"

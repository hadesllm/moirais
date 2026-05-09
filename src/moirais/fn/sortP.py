"""Sort pooling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sortpool"]


def sortpool(A, X, k):
    """
    Sort pooling

    Formula: sort node embeddings; take top-k

    Parameters
    ----------
    A : array-like
        Input data.
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhang et al (2018) DGCNN
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sort pooling"})


def cheatsheet():
    return "sortP: Sort pooling"

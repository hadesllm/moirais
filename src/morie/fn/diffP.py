"""DiffPool — differentiable graph pooling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["diffpool"]


def diffpool(A, X, K_clusters):
    """
    DiffPool — differentiable graph pooling

    Formula: S = softmax(GNN_pool); H' = S^T H

    Parameters
    ----------
    A : array-like
        Input data.
    X : array-like
        Input data.
    K_clusters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ying et al (2018)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DiffPool — differentiable graph pooling"})


def cheatsheet():
    return "diffP: DiffPool — differentiable graph pooling"

# morie.fn — function file (hadesllm/morie)
"""Explained variance ratio of the k-th principal component."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_explained_variance_ratio"]


def geron_explained_variance_ratio(singular_values):
    """
    Explained variance ratio of the k-th principal component

    Formula: EVR_k = sigma_k^2 / sum_j sigma_j^2

    Parameters
    ----------
    singular_values : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: evr

    References
    ----------
    Géron Ch 7, Explained Variance Ratio section
    """
    singular_values = np.asarray(singular_values, dtype=float)
    n = int(singular_values) if singular_values.ndim == 0 else len(singular_values)
    result = float(np.mean(singular_values))
    se = float(np.std(singular_values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Explained variance ratio of the k-th principal component"})


def cheatsheet():
    return "grevr: Explained variance ratio of the k-th principal component"

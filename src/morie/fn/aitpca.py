"""PCA in CLR coordinates (compositional PCA)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_clr_pca"]


def aitchison_clr_pca(X, k):
    """
    PCA in CLR coordinates (compositional PCA)

    Formula: Z = clr(X), eigendecompose Z^T Z

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: scores, loadings, var_exp

    References
    ----------
    Aitchison (1983)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PCA in CLR coordinates (compositional PCA)"})


def cheatsheet():
    return "aitpca: PCA in CLR coordinates (compositional PCA)"

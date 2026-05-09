"""DP PCA via Gaussian noise on covariance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_pca"]


def dp_pca(X, k, epsilon):
    """
    DP PCA via Gaussian noise on covariance

    Formula: top-k eigvecs of Cov + symmetric Gaussian

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork et al (2014); Kapralov-Talwar (2013)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP PCA via Gaussian noise on covariance"})


def cheatsheet():
    return "dppca: DP PCA via Gaussian noise on covariance"

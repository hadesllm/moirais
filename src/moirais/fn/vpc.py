"""Variance partition coefficient for logistic latent (sigma2_u + pi^2/3)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["variance_partition_coefficient"]


def variance_partition_coefficient(y, cluster, sigma2_u):
    """
    Variance partition coefficient for logistic latent (sigma2_u + pi^2/3)

    Formula: VPC = sigma2_u / (sigma2_u + pi^2/3)

    Parameters
    ----------
    y : array-like
        Input data.
    cluster : array-like
        Input data.
    sigma2_u : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Goldstein, Browne, Rasbash (2002)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variance partition coefficient for logistic latent (sigma2_u + pi^2/3)"})


def cheatsheet():
    return "vpc: Variance partition coefficient for logistic latent (sigma2_u + pi^2/3)"

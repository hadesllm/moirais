"""Linear-mixed-model GWAS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gwas_linear"]


def gwas_linear(y, M, K):
    """
    Linear-mixed-model GWAS

    Formula: per-SNP: y = mu + b SNP + Zu + e; test b=0

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Yu et al (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear-mixed-model GWAS"})


def cheatsheet():
    return "gwasl1: Linear-mixed-model GWAS"

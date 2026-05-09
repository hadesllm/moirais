"""Centering within cluster mean (CWC) for level-1 covariate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["centering_within_cluster_mean"]


def centering_within_cluster_mean(y, cluster):
    """
    Centering within cluster mean (CWC) for level-1 covariate

    Formula: x_ij_CWC = x_ij - xbar_j

    Parameters
    ----------
    y : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Enders & Tofighi (2007)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Centering within cluster mean (CWC) for level-1 covariate"})


def cheatsheet():
    return "cwcm: Centering within cluster mean (CWC) for level-1 covariate"

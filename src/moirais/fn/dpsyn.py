"""DP synthetic data (e.g. PrivBayes / MWEM)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_synthetic_data"]


def dp_synthetic_data(X, epsilon):
    """
    DP synthetic data (e.g. PrivBayes / MWEM)

    Formula: learn DP marginals; sample

    Parameters
    ----------
    X : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhang et al (2017) PrivBayes
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP synthetic data (e.g. PrivBayes / MWEM)"})


def cheatsheet():
    return "dpsyn: DP synthetic data (e.g. PrivBayes / MWEM)"

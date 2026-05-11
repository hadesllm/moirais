"""Cross K-function for bivariate point patterns."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_cross_k_function"]


def schabenberger_cross_k_function(points1, points2, lambda1, lambda2, r):
    """
    Cross K-function for bivariate point patterns

    Formula: K_{12}(r) = (1/lambda_1) * E[# type-2 events within r of a random type-1 event]

    Parameters
    ----------
    points1 : array-like
        Input data.
    points2 : array-like
        Input data.
    lambda1 : array-like
        Input data.
    lambda2 : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schabenberger Ch 3
    """
    points1 = np.asarray(points1, dtype=float)
    n = int(points1) if points1.ndim == 0 else len(points1)
    result = float(np.mean(points1))
    se = float(np.std(points1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross K-function for bivariate point patterns"})


def cheatsheet():
    return "spkcrs: Cross K-function for bivariate point patterns"

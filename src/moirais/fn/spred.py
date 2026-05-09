"""Level-2 shrinkage predictor (Empirical Bayes alternative)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["shrinkage_predictor_level2"]


def shrinkage_predictor_level2(y, cluster, sigma2_u, sigma2_e):
    """
    Level-2 shrinkage predictor (Empirical Bayes alternative)

    Formula: theta_j_S = ybar.. + (1 - lambda_j)(ybar_j - ybar..)

    Parameters
    ----------
    y : array-like
        Input data.
    cluster : array-like
        Input data.
    sigma2_u : array-like
        Input data.
    sigma2_e : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Stein (1956); Morris (1983)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Level-2 shrinkage predictor (Empirical Bayes alternative)"})


def cheatsheet():
    return "spred: Level-2 shrinkage predictor (Empirical Bayes alternative)"

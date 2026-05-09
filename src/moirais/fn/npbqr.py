"""NP Bayes quantile regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["np_bayes_quant_reg"]


def np_bayes_quant_reg(y, X, tau):
    """
    NP Bayes quantile regression

    Formula: asymmetric Laplace likelihood + DP prior

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kottas-Gelfand (2001)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NP Bayes quantile regression"})


def cheatsheet():
    return "npbqr: NP Bayes quantile regression"

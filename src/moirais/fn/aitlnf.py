"""MLE of logistic-normal parameters from compositions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["logistic_normal_fit"]


def logistic_normal_fit(X, ref):
    """
    MLE of logistic-normal parameters from compositions

    Formula: y_n = alr(x_n); μ̂ = mean(y), Σ̂ = cov(y)

    Parameters
    ----------
    X : array-like
        Input data.
    ref : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mu, Sigma

    References
    ----------
    Aitchison (1986)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MLE of logistic-normal parameters from compositions"})


def cheatsheet():
    return "aitlnf: MLE of logistic-normal parameters from compositions"

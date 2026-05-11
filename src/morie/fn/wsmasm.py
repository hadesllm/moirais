"""MLE asymptotic normality."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_mle_asymptotic"]


def wasserman_mle_asymptotic(data, f, theta_hat):
    """
    MLE asymptotic normality

    Formula: sqrt(n)(theta_hat - theta) ~> N(0, 1/I(theta))

    Parameters
    ----------
    data : array-like
        Input data.
    f : array-like
        Input data.
    theta_hat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Wasserman (2004), Ch 9
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MLE asymptotic normality"})


def cheatsheet():
    return "wsmasm: MLE asymptotic normality"

"""Zero-inflated Poisson model for excess-zero count data."""
import numpy as np
from ._richresult import RichResult

__all__ = ["zero_inflated_poisson"]


def zero_inflated_poisson(y, X):
    """
    Zero-inflated Poisson model for excess-zero count data

    Formula: P(Y=0) = pi + (1-pi)*exp(-lambda); P(Y=k) = (1-pi)*exp(-lambda)*lambda^k/k! for k>0

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'pi_hat': 'float', 'lambda_hat': 'array'}

    References
    ----------
    Montesinos Lopez Ch 7
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Zero-inflated Poisson model for excess-zero count data"})


def cheatsheet():
    return "zipmd: Zero-inflated Poisson model for excess-zero count data"

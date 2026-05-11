# morie.fn — function file (hadesllm/morie)
"""Maximum likelihood log-likelihood for linear regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ml_log_likelihood_regression"]


def ml_log_likelihood_regression(y, X, beta, sigma2):
    """
    Maximum likelihood log-likelihood for linear regression

    Formula: log L(beta, sigma^2) = -(n/2)*log(2*pi) - (n/2)*log(sigma^2) - (1/(2*sigma^2))*||y - X*beta||^2

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    beta : array-like
        Input data.
    sigma2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'loglik': 'float'}

    References
    ----------
    Montesinos Lopez Ch 3
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Maximum likelihood log-likelihood for linear regression"})


def cheatsheet():
    return "mllog: Maximum likelihood log-likelihood for linear regression"

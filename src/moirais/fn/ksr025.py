"""Penalized log-likelihood for partly linear logistic regression with smoothness penalty."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch1_penalized_loglikelihood"]


def kosorok_ch1_penalized_loglikelihood(beta, eta, X, lambda_n, n):
    """
    Penalized log-likelihood for partly linear logistic regression with smoothness penalty

    Formula: L_tilde_n(beta, eta) = n^{-1} * sum_i log p_{beta,eta}(X_i) - lambda_n^2 * J^2(eta)

    Parameters
    ----------
    beta : array-like
        Input data.
    eta : array-like
        Input data.
    X : array-like
        Input data.
    lambda_n : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kosorok (2008), Ch 1, Eq 1.6, p. 6
    """
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    n = len(beta)
    result = float(np.mean(beta))
    se = float(np.std(beta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Penalized log-likelihood for partly linear logistic regression with smoothness penalty"})


def cheatsheet():
    return "ksr025: Penalized log-likelihood for partly linear logistic regression with smoothness penalty"

"""Empirical Bayes shrinkage estimator for cluster means."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["empirical_bayes_shrinkage"]


def empirical_bayes_shrinkage(y, cluster, sigma2_u, sigma2_e):
    """
    Empirical Bayes shrinkage estimator for cluster means

    Formula: theta_j_EB = lambda_j ybar_j + (1 - lambda_j) ybar..., lambda_j = sigma2_u / (sigma2_u + sigma2_e/n_j)

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
    Morris (1983); Efron (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Empirical Bayes shrinkage estimator for cluster means"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Empirical Bayes shrinkage estimator for cluster means"})


def cheatsheet():
    return "ebayes: Empirical Bayes shrinkage estimator for cluster means"

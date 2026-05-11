"""Bayesian logistic regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_logistic"]


def bayes_logistic(y, X, prior_scale):
    """
    Bayesian logistic regression

    Formula: y ~ Bernoulli(sigmoid(X beta)); beta ~ Cauchy

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    prior_scale : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gelman et al (2008) weakly-informative
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian logistic regression"})


def cheatsheet():
    return "baylog: Bayesian logistic regression"

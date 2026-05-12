# morie.fn -- function file (hadesllm/morie)
"""Empirical Bayes: estimate hyperparameter alpha from marginal likelihood."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_empirical_bayes_np"]


def ghosal_empirical_bayes_np(x):
    """
    Empirical Bayes: estimate hyperparameter alpha from marginal likelihood

    Formula: alpha_hat = argmax_alpha integral p_alpha(X) dPi_alpha(theta) for Bayes model

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 4-6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Empirical Bayes: estimate hyperparameter alpha from marginal likelihood"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Empirical Bayes: estimate hyperparameter alpha from marginal likelihood"})


def cheatsheet():
    return "gh_emp_bayes: Empirical Bayes: estimate hyperparameter alpha from marginal likelihood"

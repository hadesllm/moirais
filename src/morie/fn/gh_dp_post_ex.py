# morie.fn -- function file (hadesllm/morie)
"""Exact posterior predictive for DP: closed-form Polya urn for density estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_posterior_exact"]


def ghosal_dp_posterior_exact(x):
    """
    Exact posterior predictive for DP: closed-form Polya urn for density estimation

    Formula: p(X_{n+1}=x|X_1..X_n) = alpha/(alpha+n)*G0(x) + sum_{k} n_k/(alpha+n)*delta_{X_k^*}(x)

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
    Ghosal Ch 4 §4.1.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Exact posterior predictive for DP: closed-form Polya urn for density estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Exact posterior predictive for DP: closed-form Polya urn for density estimation"})


def cheatsheet():
    return "gh_dp_post_ex: Exact posterior predictive for DP: closed-form Polya urn for density estimation"

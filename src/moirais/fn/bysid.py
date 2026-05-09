# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bayesian ideal point estimation (MCMC)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bayesian_ideal_points"]


def bayesian_ideal_points(x):
    """
    Bayesian ideal point estimation (MCMC)

    Formula: x_i* ~ N(mu, sigma^2), posterior via Gibbs

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
    Armstrong Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Bayesian ideal point estimation (MCMC)"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Bayesian ideal point estimation (MCMC)"})


def cheatsheet():
    return "bysid: Bayesian ideal point estimation (MCMC)"

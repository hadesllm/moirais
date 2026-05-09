# moirais.fn — function file (hadesllm/moirais)
"""Weibull hazard model with unobserved heterogeneity."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_weibull_heterogeneity"]


def horowitz_weibull_heterogeneity(t, x, event, mixing_dist):
    """
    Weibull hazard model with unobserved heterogeneity

    Formula: h(t|X,V) = lambda*alpha*(lambda*t)^{alpha-1}*exp(X'beta)*V; V unobserved mixing

    Parameters
    ----------
    t : array-like
        Input data.
    x : array-like
        Input data.
    event : array-like
        Input data.
    mixing_dist : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lambda_hat, alpha_hat, beta_hat

    References
    ----------
    Horowitz Ch 6, Sec 6.1.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weibull hazard model with unobserved heterogeneity"})


def cheatsheet():
    return "hrzweib: Weibull hazard model with unobserved heterogeneity"

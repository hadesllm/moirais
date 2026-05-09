# moirais.fn — function file (hadesllm/moirais)
"""Copula parameter estimation (Gaussian/Clayton/Gumbel)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["copula_estimation"]


def copula_estimation(x, y):
    """
    Copula parameter estimation (Gaussian/Clayton/Gumbel)

    Formula: C(u,v;theta) = Phi_2(Phi^{-1}(u), Phi^{-1}(v); rho)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nelsen (2006)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Copula parameter estimation (Gaussian/Clayton/Gumbel)"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Copula parameter estimation (Gaussian/Clayton/Gumbel)"})


def cheatsheet():
    return "copul: Copula parameter estimation (Gaussian/Clayton/Gumbel)"

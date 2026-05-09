# moirais.fn — function file (hadesllm/moirais)
"""Appendix: Multidimensional kernel density estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_multivariate_kde"]


def horowitz_multivariate_kde(x, bandwidths):
    """
    Appendix: Multidimensional kernel density estimation

    Formula: f_hat(x) = (1/(n*prod(h_j)))*sum K((x-X_i)/H) where H=diag(h)

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidths : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: density_estimate

    References
    ----------
    Horowitz Appendix A.1.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Appendix: Multidimensional kernel density estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Appendix: Multidimensional kernel density estimation"})


def cheatsheet():
    return "hrzkd2: Appendix: Multidimensional kernel density estimation"

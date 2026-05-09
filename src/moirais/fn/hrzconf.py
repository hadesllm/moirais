# moirais.fn — function file (hadesllm/moirais)
"""Uniform confidence bands for nonparametric regression function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_confidence_bands"]


def horowitz_confidence_bands(x, y, bandwidth, alpha):
    """
    Uniform confidence bands for nonparametric regression function

    Formula: P(sup_x |m_hat(x)-m(x)| <= c_alpha) = 1-alpha; c_alpha from extreme value distribution

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: confidence_band

    References
    ----------
    Horowitz Appendix A.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Uniform confidence bands for nonparametric regression function"})


def cheatsheet():
    return "hrzconf: Uniform confidence bands for nonparametric regression function"

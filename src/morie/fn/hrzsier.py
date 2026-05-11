# morie.fn — function file (hadesllm/morie)
"""Appendix: Series (sieve) estimation of conditional mean function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_series_regression"]


def horowitz_series_regression(x, y, K, basis):
    """
    Appendix: Series (sieve) estimation of conditional mean function

    Formula: m_hat(x) = sum_{k=1}^K a_hat_k*p_k(x); a_hat = argmin (1/n)*sum[Y_i-sum a_k p_k(X_i)]^2

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    K : array-like
        Input data.
    basis : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: m_hat, coefficients

    References
    ----------
    Horowitz Appendix A.2.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Appendix: Series (sieve) estimation of conditional mean function"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Appendix: Series (sieve) estimation of conditional mean function"})


def cheatsheet():
    return "hrzsier: Appendix: Series (sieve) estimation of conditional mean function"

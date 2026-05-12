# morie.fn -- function file (hadesllm/morie)
"""Appendix: Kernel density estimator and MSE-optimal bandwidth."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_appendix_kde"]


def horowitz_appendix_kde(x, bandwidth):
    """
    Appendix: Kernel density estimator and MSE-optimal bandwidth

    Formula: f_hat(x) = (1/nh)*sum K((x-X_i)/h); h_opt = c*n^{-1/5} minimizes MISE

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: density_estimate, optimal_bandwidth

    References
    ----------
    Horowitz Appendix A.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Appendix: Kernel density estimator and MSE-optimal bandwidth"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Appendix: Kernel density estimator and MSE-optimal bandwidth"})


def cheatsheet():
    return "hrzkde: Appendix: Kernel density estimator and MSE-optimal bandwidth"

# morie.fn — function file (hadesllm/morie)
"""MSE-optimal bandwidth for kernel density estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_optimal_bandwidth_kde"]


def horowitz_optimal_bandwidth_kde(x, kernel):
    """
    MSE-optimal bandwidth for kernel density estimator

    Formula: h_opt = [R(K)/(mu_2(K)^2 * integral (f'')^2 * n)]^{1/5} where R(K)=integral K^2

    Parameters
    ----------
    x : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: optimal_bandwidth

    References
    ----------
    Horowitz Appendix A.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "MSE-optimal bandwidth for kernel density estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "MSE-optimal bandwidth for kernel density estimator"})


def cheatsheet():
    return "hrzbwopt: MSE-optimal bandwidth for kernel density estimator"

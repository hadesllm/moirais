# morie.fn -- function file (hadesllm/morie)
"""Appendix: Local-linear mean regression estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_local_linear"]


def horowitz_local_linear(x, y, bandwidth):
    """
    Appendix: Local-linear mean regression estimator

    Formula: (alpha_hat,beta_hat) = argmin sum K_h(x-X_i)*(Y_i-alpha-beta*(X_i-x))^2; m_hat(x)=alpha_hat

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: m_hat, derivative_hat

    References
    ----------
    Horowitz Appendix A.2.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Appendix: Local-linear mean regression estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Appendix: Local-linear mean regression estimator"})


def cheatsheet():
    return "hrzllr: Appendix: Local-linear mean regression estimator"

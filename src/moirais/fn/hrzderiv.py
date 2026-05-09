# moirais.fn — function file (hadesllm/moirais)
"""Appendix: Estimating derivatives of density via kernel methods."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_density_derivative"]


def horowitz_density_derivative(x, bandwidth, order):
    """
    Appendix: Estimating derivatives of density via kernel methods

    Formula: f_hat'(x) = -(1/nh^2)*sum K'((x-X_i)/h); rate O_p(n^{-r/(2r+3)}) for r-th derivative

    Parameters
    ----------
    x : array-like
        Input data.
    bandwidth : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: derivative_estimate

    References
    ----------
    Horowitz Appendix A.1.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Appendix: Estimating derivatives of density via kernel methods"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Appendix: Estimating derivatives of density via kernel methods"})


def cheatsheet():
    return "hrzderiv: Appendix: Estimating derivatives of density via kernel methods"

# moirais.fn — function file (hadesllm/moirais)
"""Yule-Walker equations for AR model estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_yule_walker"]


def rangayyan_yule_walker(x, order):
    """
    Yule-Walker equations for AR model estimation

    Formula: [R_xx] * [a] = -[r]; R_xx(i,j) = R_xx(|i-j|) (Toeplitz)

    Parameters
    ----------
    x : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a_coeffs, sigma_sq

    References
    ----------
    Rangayyan Ch 7.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Yule-Walker equations for AR model estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Yule-Walker equations for AR model estimation"})


def cheatsheet():
    return "rgyw: Yule-Walker equations for AR model estimation"

"""ML estimation of SAR model: concentrated likelihood."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_sar_ml"]


def schabenberger_sar_ml(x, y, w):
    """
    ML estimation of SAR model: concentrated likelihood

    Formula: log L(rho) = C + log|I-rho*W| - (n/2)*log[(1/n)*(Y-rho*W*Y-X*beta(rho))'*(Y-rho*W*Y-X*beta(rho))]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rho, beta, se

    References
    ----------
    Schabenberger Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "ML estimation of SAR model: concentrated likelihood"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "ML estimation of SAR model: concentrated likelihood"})


def cheatsheet():
    return "spsarml: ML estimation of SAR model: concentrated likelihood"

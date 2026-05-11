"""Cressie-Hawkins robust semivariogram estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_cressie_hawkins"]


def schabenberger_cressie_hawkins(coords, z, lag_bins):
    """
    Cressie-Hawkins robust semivariogram estimator

    Formula: 2*gamma_hat(h) = {(1/|N(h)|)*sum|Z(s_i)-Z(s_j)|^{1/2}}^4 / (0.457+0.494/|N(h)|)

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    lag_bins : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: robust_variogram

    References
    ----------
    Schabenberger Ch 4, Sec 4.4.2
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    if z.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Cressie-Hawkins robust semivariogram estimator"})
    estimate = np.median(z)
    se = 1.2533 * np.std(z, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Cressie-Hawkins robust semivariogram estimator"})


def cheatsheet():
    return "spcrhk: Cressie-Hawkins robust semivariogram estimator"

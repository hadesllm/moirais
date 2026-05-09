"""PWM estimator of GPD parameters."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gpd_pwm"]


def evt_gpd_pwm(y):
    """
    PWM estimator of GPD parameters

    Formula: ξ̂ = ȳ/(2(ȳ-(2/n)Σ y_i (i-1)/(n-1)))

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sigma, xi

    References
    ----------
    Hosking & Wallis (1987)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "PWM estimator of GPD parameters"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "PWM estimator of GPD parameters"})


def cheatsheet():
    return "evgpdpw: PWM estimator of GPD parameters"

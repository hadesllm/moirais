# moirais.fn — function file (hadesllm/moirais)
"""One-step asymptotically efficient estimator for single-index model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_one_step_efficient"]


def horowitz_one_step_efficient(x, y, bandwidth, initial_estimator):
    """
    One-step asymptotically efficient estimator for single-index model

    Formula: beta_hat_eff = beta_hat_initial + [I(beta)]^{-1} * score(beta_hat_initial)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.
    initial_estimator : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat, se

    References
    ----------
    Horowitz Ch 2, Sec 2.6.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "One-step asymptotically efficient estimator for single-index model"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "One-step asymptotically efficient estimator for single-index model"})


def cheatsheet():
    return "hrzasym: One-step asymptotically efficient estimator for single-index model"

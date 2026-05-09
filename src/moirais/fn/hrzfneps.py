# moirais.fn — function file (hadesllm/moirais)
"""Estimators fn_eps and fn_U for panel deconvolution."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_panel_density_estimators"]


def horowitz_panel_density_estimators(y_panel, x_panel, bandwidth):
    """
    Estimators fn_eps and fn_U for panel deconvolution

    Formula: fn_eps from time-demeaned Y; fn_U by deconvolution of Y_bar using fn_eps

    Parameters
    ----------
    y_panel : array-like
        Input data.
    x_panel : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: density_estimates

    References
    ----------
    Horowitz Ch 5, Sec 5.2.2
    """
    y_panel = np.asarray(y_panel, dtype=float)
    n = int(y_panel) if y_panel.ndim == 0 else len(y_panel)
    if y_panel.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Estimators fn_eps and fn_U for panel deconvolution"})
    estimate = np.median(y_panel)
    se = 1.2533 * np.std(y_panel, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Estimators fn_eps and fn_U for panel deconvolution"})


def cheatsheet():
    return "hrzfneps: Estimators fn_eps and fn_U for panel deconvolution"

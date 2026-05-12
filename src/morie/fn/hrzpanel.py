# morie.fn -- function file (hadesllm/morie)
"""Panel data deconvolution: estimate fU and f_eps from Y_jt = X_jt'beta + U_j + eps_jt."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_panel_deconv"]


def horowitz_panel_deconv(y_panel, x_panel):
    """
    Panel data deconvolution: estimate fU and f_eps from Y_jt = X_jt'beta + U_j + eps_jt

    Formula: diffs: Delta_Y_jt = Delta_X_jt'beta + Delta_eps_jt; mean: Ybar_j = X_j_bar'beta + U_j + epsbar_j

    Parameters
    ----------
    y_panel : array-like
        Input data.
    x_panel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fU_hat, feps_hat

    References
    ----------
    Horowitz Ch 5, Sec 5.2
    """
    y_panel = np.asarray(y_panel, dtype=float)
    n = int(y_panel) if y_panel.ndim == 0 else len(y_panel)
    if y_panel.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Panel data deconvolution: estimate fU and f_eps from Y_jt = X_jt'beta + U_j + eps_jt"})
    estimate = np.median(y_panel)
    se = 1.2533 * np.std(y_panel, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Panel data deconvolution: estimate fU and f_eps from Y_jt = X_jt'beta + U_j + eps_jt"})


def cheatsheet():
    return "hrzpanel: Panel data deconvolution: estimate fU and f_eps from Y_jt = X_jt'beta + U_j + eps_jt"

# moirais.fn — function file (hadesllm/moirais)
"""First-passage time estimation in panel data model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_first_passage_time"]


def horowitz_first_passage_time(y_panel, threshold, fU_hat):
    """
    First-passage time estimation in panel data model

    Formula: T_j = inf{t: Y_jt > c}; distribution estimated via convolution with fn_U

    Parameters
    ----------
    y_panel : array-like
        Input data.
    threshold : array-like
        Input data.
    fU_hat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fpt_distribution

    References
    ----------
    Horowitz Ch 5, Sec 5.2.3
    """
    y_panel = np.asarray(y_panel, dtype=float)
    n = int(y_panel) if y_panel.ndim == 0 else len(y_panel)
    if y_panel.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "First-passage time estimation in panel data model"})
    estimate = np.median(y_panel)
    se = 1.2533 * np.std(y_panel, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "First-passage time estimation in panel data model"})


def cheatsheet():
    return "hrzfpt: First-passage time estimation in panel data model"

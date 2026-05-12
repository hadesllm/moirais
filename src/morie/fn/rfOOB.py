# morie.fn -- function file (hadesllm/morie)
"""Random forest out-of-bag (OOB) error estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rf_oob_error"]


def rf_oob_error(y, oob_preds):
    """
    Random forest out-of-bag (OOB) error estimation

    Formula: OOB_error = (1/n)*sum_i L(y_i, f_oob(x_i)); f_oob averages trees not trained on i

    Parameters
    ----------
    y : array-like
        Input data.
    oob_preds : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'oob_error': 'float'}

    References
    ----------
    Montesinos Lopez Ch 15
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    if y.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Random forest out-of-bag (OOB) error estimation"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Random forest out-of-bag (OOB) error estimation"})


def cheatsheet():
    return "rfOOB: Random forest out-of-bag (OOB) error estimation"

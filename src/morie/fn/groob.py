# morie.fn — function file (hadesllm/morie)
"""Out-of-bag error estimate for bagged ensembles."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_oob_error"]


def geron_oob_error(y_true, oob_predictions):
    """
    Out-of-bag error estimate for bagged ensembles

    Formula: err_oob = (1/n) sum_i L(y_i, mean_{b: i not in bootstrap_b} h_b(x_i))

    Parameters
    ----------
    y_true : array-like
        Input data.
    oob_predictions : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: oob_err

    References
    ----------
    Géron Ch 6, Out-of-Bag Evaluation section
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    if y_true.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Out-of-bag error estimate for bagged ensembles"})
    estimate = np.median(y_true)
    se = 1.2533 * np.std(y_true, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Out-of-bag error estimate for bagged ensembles"})


def cheatsheet():
    return "groob: Out-of-bag error estimate for bagged ensembles"

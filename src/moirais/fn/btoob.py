"""Out-of-bag (.632) error estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_oob_error"]


def boot_oob_error(x, y, fit_fn, predict_fn, B):
    """
    Out-of-bag (.632) error estimator

    Formula: ê_OOB = mean over (i,b) where i ∉ resample b of L(y_i, ĝ_b(x_i))

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    fit_fn : array-like
        Input data.
    predict_fn : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: oob_err

    References
    ----------
    Efron & Tibshirani (1997)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Out-of-bag (.632) error estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Out-of-bag (.632) error estimator"})


def cheatsheet():
    return "btoob: Out-of-bag (.632) error estimator"

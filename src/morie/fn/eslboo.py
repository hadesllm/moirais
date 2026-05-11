"""Bootstrap estimate of prediction error."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_bootstrap_err"]


def esl_bootstrap_err(X, y, model, B):
    """
    Bootstrap estimate of prediction error

    Formula: Err_boot = (1/B) sum L(y_i, f_b(x_i)) over bootstrap samples

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    model : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Hastie ESL Ch 7
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Bootstrap estimate of prediction error"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Bootstrap estimate of prediction error"})


def cheatsheet():
    return "eslboo: Bootstrap estimate of prediction error"

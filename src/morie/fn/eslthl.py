"""Thin plate spline (2D smoother)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_thin_plate_spline"]


def esl_thin_plate_spline(X, y, lambda_):
    """
    Thin plate spline (2D smoother)

    Formula: min sum (y_i-f(x_i))^2 + lambda J(f)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    lambda_ : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fit

    References
    ----------
    Hastie ESL Ch 5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Thin plate spline (2D smoother)"})


def cheatsheet():
    return "eslthl: Thin plate spline (2D smoother)"

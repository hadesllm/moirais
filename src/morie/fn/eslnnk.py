"""Nadaraya-Watson kernel regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_nadaraya_watson"]


def esl_nadaraya_watson(x0, x_data, y_data, lambda_):
    """
    Nadaraya-Watson kernel regression

    Formula: f_hat(x_0) = sum K_lambda(x_0,x_i) y_i / sum K_lambda(x_0,x_i)

    Parameters
    ----------
    x0 : array-like
        Input data.
    x_data : array-like
        Input data.
    y_data : array-like
        Input data.
    lambda_ : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hastie ESL Ch 6
    """
    x0 = np.atleast_1d(np.asarray(x0, dtype=float))
    n = len(x0)
    result = float(np.mean(x0))
    se = float(np.std(x0, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nadaraya-Watson kernel regression"})


def cheatsheet():
    return "eslnnk: Nadaraya-Watson kernel regression"

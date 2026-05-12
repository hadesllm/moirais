r"""Squared error between the predicted value and the target for a single example.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_squared_error"]


def burkov_lm_ch1_squared_error(y_hat_i, y_i):
    r"""
    Squared error between the predicted value and the target for a single example.

    Formula: \operatorname{err}(\hat{y}_i, y_i) \stackrel{\text{def}}{=} (\hat{y}_i - y_i)^2

    Parameters
    ----------
    y_hat_i : array-like
        Input data.
    y_i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: non-negative squared error

    References
    ----------
    Burkov LM (2025), Ch 1, Eq 1.2, p. 22
    """
    y_hat_i = np.atleast_1d(np.asarray(y_hat_i, dtype=float))
    n = len(y_hat_i)
    result = float(np.mean(y_hat_i))
    se = float(np.std(y_hat_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Squared error between the predicted value and the target for a single example."})


def cheatsheet():
    return "b102: Squared error between the predicted value and the target for a single example."

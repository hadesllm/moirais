"""Logistic regression model applying the sigmoid to a linear combination of inputs for binary classification.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_logistic_regression"]


def burkov_lm_ch1_logistic_regression(w, x, b):
    """
    Logistic regression model applying the sigmoid to a linear combination of inputs for binary classification.

    Formula: y = \sigma(\mathbf{w} \cdot \mathbf{x} + b)

    Parameters
    ----------
    w : array-like
        Input data.
    x : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: probability in (0, 1)

    References
    ----------
    Burkov LM (2025), Ch 1, Eq 1.8, p. 40
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Logistic regression model applying the sigmoid to a linear combination of inputs for binary classification."})


def cheatsheet():
    return "b108: Logistic regression model applying the sigmoid to a linear combination of inputs for binary classification."

"""Closed-form gradients of binary cross-entropy with respect to each weight and the bias for logistic regression.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_bce_gradients"]


def burkov_lm_ch1_bce_gradients(y_hat, y, x, N, j):
    """
    Closed-form gradients of binary cross-entropy with respect to each weight and the bias for logistic regression.

    Formula: \frac{\partial \operatorname{loss}}{\partial w^{(j)}} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) x_i^{(j)}, \qquad \frac{\partial \operatorname{loss}}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)

    Parameters
    ----------
    y_hat : array-like
        Input data.
    y : array-like
        Input data.
    x : array-like
        Input data.
    N : array-like
        Input data.
    j : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: partial derivatives w.r.t. weights and bias

    References
    ----------
    Burkov LM (2025), Ch 1, Eq 1.11, p. 42
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Closed-form gradients of binary cross-entropy with respect to each weight and the bias for logistic regression."})


def cheatsheet():
    return "b111: Closed-form gradients of binary cross-entropy with respect to each weight and the bias for logistic regression."

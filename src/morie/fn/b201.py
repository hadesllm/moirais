r"""Simplified categorical cross-entropy loss when the target is one-hot, reducing to the negative log-probability of the correct class c.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch2_categorical_cross_entropy"]


def burkov_lm_ch2_categorical_cross_entropy(y_hat, c):
    r"""
    Simplified categorical cross-entropy loss when the target is one-hot, reducing to the negative log-probability of the correct class c.

    Formula: \operatorname{loss}(\hat{\mathbf{y}}, \mathbf{y}) = -\log\!\bigl(\hat{y}^{(c)}\bigr)

    Parameters
    ----------
    y_hat : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: negative log-likelihood of correct class

    References
    ----------
    Burkov LM (2025), Ch 2, Eq 2.1, p. 57
    """
    y_hat = np.atleast_1d(np.asarray(y_hat, dtype=float))
    n = len(y_hat)
    result = float(np.mean(y_hat))
    se = float(np.std(y_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simplified categorical cross-entropy loss when the target is one-hot, reducing to the negative log-probability of the correct class c."})


def cheatsheet():
    return "b201: Simplified categorical cross-entropy loss when the target is one-hot, reducing to the negative log-probability of the correct class c."

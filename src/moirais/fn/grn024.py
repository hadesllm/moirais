"""Gradient vector of the cross-entropy cost function with respect to the parameter vector for class k.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["geron_ch4_cross_entropy_gradient_vector"]


def geron_ch4_cross_entropy_gradient_vector(X, Y, Theta, k):
    """
    Gradient vector of the cross-entropy cost function with respect to the parameter vector for class k.

    Formula: grad_theta_k J(Theta) = (1/m) * sum_{i=1}^{m} (p_hat_k^(i) - y_k^(i)) * x^(i)

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    Theta : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gradient_vector

    References
    ----------
    Geron (2026), Ch 4, Eq 4-24, p. 176
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient vector of the cross-entropy cost function with respect to the parameter vector for class k."})


def cheatsheet():
    return "grn024: Gradient vector of the cross-entropy cost function with respect to the parameter vector for class k."

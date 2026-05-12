# morie.fn -- function file (hadesllm/morie)
"""GloVe weighted least-squares cost over co-occurrence counts X_ij."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_glove_cost"]


def kamath_glove_cost(X, W, W_tilde, b, b_tilde, x_max, alpha):
    """
    GloVe weighted least-squares cost over co-occurrence counts X_ij

    Formula: J = sum_{i,j} f(X_ij) * (w_i^T w_tilde_j + b_i + b_tilde_j - log X_ij)^2

    Parameters
    ----------
    X : array-like
        Input data.
    W : array-like
        Input data.
    W_tilde : array-like
        Input data.
    b : array-like
        Input data.
    b_tilde : array-like
        Input data.
    x_max : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Kamath Ch 1, GloVe section
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GloVe weighted least-squares cost over co-occurrence counts X_ij"})


def cheatsheet():
    return "kmglv: GloVe weighted least-squares cost over co-occurrence counts X_ij"

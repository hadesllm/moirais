# moirais.fn — function file (hadesllm/moirais)
"""L1 regularization adds |theta| penalty to cost."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_l1_regularization"]


def geron_l1_regularization(theta, alpha):
    """
    L1 regularization adds |theta| penalty to cost

    Formula: J = J_data + alpha * sum |theta_i|

    Parameters
    ----------
    theta : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: penalty

    References
    ----------
    Géron Ch 11
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "L1 regularization adds |theta| penalty to cost"})


def cheatsheet():
    return "hml1r: L1 regularization adds |theta| penalty to cost"

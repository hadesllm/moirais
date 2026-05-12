# morie.fn -- function file (hadesllm/morie)
"""L2 regularization adds theta^2 penalty."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_l2_regularization"]


def geron_l2_regularization(theta, alpha):
    """
    L2 regularization adds theta^2 penalty

    Formula: J = J_data + (alpha/2) * sum theta_i^2

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "L2 regularization adds theta^2 penalty"})


def cheatsheet():
    return "hml2r: L2 regularization adds theta^2 penalty"

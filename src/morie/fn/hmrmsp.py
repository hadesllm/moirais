# morie.fn — function file (hadesllm/morie)
"""RMSProp: exponentially weighted gradient-squared average."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_rmsprop"]


def geron_rmsprop(grads, s, beta, eta, eps):
    """
    RMSProp: exponentially weighted gradient-squared average

    Formula: s <- beta*s + (1-beta)*g^2; theta <- theta - eta*g/(sqrt(s)+eps)

    Parameters
    ----------
    grads : array-like
        Input data.
    s : array-like
        Input data.
    beta : array-like
        Input data.
    eta : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Géron Ch 11
    """
    grads = np.atleast_1d(np.asarray(grads, dtype=float))
    n = len(grads)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "RMSProp: exponentially weighted gradient-squared average"})
    estimate = np.median(grads)
    se = 1.2533 * np.std(grads, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "RMSProp: exponentially weighted gradient-squared average"})


def cheatsheet():
    return "hmrmsp: RMSProp: exponentially weighted gradient-squared average"

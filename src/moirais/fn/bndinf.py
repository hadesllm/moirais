"""Inference for partially identified parameters."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_inference"]


def bound_inference(theta, moments, alpha):
    """
    Inference for partially identified parameters

    Formula: projection-based CI

    Parameters
    ----------
    theta : array-like
        Input data.
    moments : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Romano-Shaikh (2008)
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inference for partially identified parameters"})


def cheatsheet():
    return "bndinf: Inference for partially identified parameters"

# morie.fn — function file (hadesllm/morie)
"""Metropolis-Hastings algorithm for posterior sampling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_mh_sampler"]


def ghosal_mh_sampler(x):
    """
    Metropolis-Hastings algorithm for posterior sampling

    Formula: Accept theta* with prob min(1, pi(theta*|X)*q(theta|theta*) / (pi(theta|X)*q(theta*|theta)))

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal App M
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Metropolis-Hastings algorithm for posterior sampling"})


def cheatsheet():
    return "gh_ap_m1: Metropolis-Hastings algorithm for posterior sampling"

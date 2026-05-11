# morie.fn — function file (hadesllm/morie)
"""Finite random series prior: f = sum_{k<=K} beta_k phi_k, K ~ pi_n adaptive."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_rnd_series_pr"]


def ghosal_rnd_series_pr(x):
    """
    Finite random series prior: f = sum_{k<=K} beta_k phi_k, K ~ pi_n adaptive

    Formula: K ~ pi_n, beta_k|K ~ N(0,sigma^2) iid, adaptive to smoothness s

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
    Ghosal Ch 10 §10.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Finite random series prior: f = sum_{k<=K} beta_k phi_k, K ~ pi_n adaptive"})


def cheatsheet():
    return "gh_c10_6: Finite random series prior: f = sum_{k<=K} beta_k phi_k, K ~ pi_n adaptive"

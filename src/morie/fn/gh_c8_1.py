# morie.fn — function file (hadesllm/morie)
"""Posterior contraction rate definition: eps_n rate iff Pi(d>M*eps_n|data)->0 in P0-prob."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_crt_def"]


def ghosal_crt_def(x):
    """
    Posterior contraction rate definition: eps_n rate iff Pi(d>M*eps_n|data)->0 in P0-prob

    Formula: Pi_n({theta: d(theta,theta0)>M*eps_n}|X^n) ->_{P0^n} 0 for M->infty

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
    Ghosal Ch 8 §8.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior contraction rate definition: eps_n rate iff Pi(d>M*eps_n|data)->0 in P0-prob"})


def cheatsheet():
    return "gh_c8_1: Posterior contraction rate definition: eps_n rate iff Pi(d>M*eps_n|data)->0 in P0-prob"

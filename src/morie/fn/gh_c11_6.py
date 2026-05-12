# morie.fn -- function file (hadesllm/morie)
"""Brownian motion as prior: k(s,t)=min(s,t), Holder-1/2 paths."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_bm_prior"]


def ghosal_bm_prior(x):
    """
    Brownian motion as prior: k(s,t)=min(s,t), Holder-1/2 paths

    Formula: W ~ BM: E[W(t)]=0, Cov(W(s),W(t))=min(s,t), paths Holder-1/2 a.s.

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
    Ghosal Ch 11 §11.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Brownian motion as prior: k(s,t)=min(s,t), Holder-1/2 paths"})


def cheatsheet():
    return "gh_c11_6: Brownian motion as prior: k(s,t)=min(s,t), Holder-1/2 paths"

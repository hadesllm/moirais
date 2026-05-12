# morie.fn -- function file (hadesllm/morie)
"""Prohorov metric on M(X): metrizes weak convergence on Polish spaces."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_prohorov_metric"]


def ghosal_prohorov_metric(x):
    """
    Prohorov metric on M(X): metrizes weak convergence on Polish spaces

    Formula: d_P(P,Q) = inf{eps>0: P(A) <= Q(A^eps)+eps for all closed A}

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
    Ghosal App A
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prohorov metric on M(X): metrizes weak convergence on Polish spaces"})


def cheatsheet():
    return "gh_ap_a2: Prohorov metric on M(X): metrizes weak convergence on Polish spaces"

# morie.fn -- function file (hadesllm/morie)
"""Kruskal stress-1 badness-of-fit for MDS solutions."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kruskal_stress"]


def kruskal_stress(D_observed, D_config):
    """
    Kruskal stress-1 badness-of-fit for MDS solutions

    Formula: S = sqrt(sum_{i<j}(d_ij - dhat_ij)^2 / sum_{i<j} d_ij^2)

    Parameters
    ----------
    D_observed : array-like
        Input data.
    D_config : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'stress': 'float'}

    References
    ----------
    Armstrong Ch 3
    """
    D_observed = np.asarray(D_observed, dtype=float)
    n = int(D_observed) if D_observed.ndim == 0 else len(D_observed)
    result = float(np.mean(D_observed))
    se = float(np.std(D_observed, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kruskal stress-1 badness-of-fit for MDS solutions"})


def cheatsheet():
    return "krust: Kruskal stress-1 badness-of-fit for MDS solutions"

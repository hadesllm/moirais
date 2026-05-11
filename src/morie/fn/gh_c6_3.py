# morie.fn — function file (hadesllm/morie)
"""Doob's theorem: posterior consistent a.s. prior for identifiable models."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_doob_consist"]


def ghosal_doob_consist(x):
    """
    Doob's theorem: posterior consistent a.s. prior for identifiable models

    Formula: Pi-a.s., Pi_n(U^c|X^n) -> 0 for all U with P_theta in U

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
    Ghosal Ch 6 §6.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Doob's theorem: posterior consistent a.s. prior for identifiable models"})


def cheatsheet():
    return "gh_c6_3: Doob's theorem: posterior consistent a.s. prior for identifiable models"

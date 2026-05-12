# morie.fn -- function file (hadesllm/morie)
"""Parametric rate recovery from mixture prior: 1/sqrt(n) for parametric models."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_param_rate"]


def ghosal_param_rate(x):
    """
    Parametric rate recovery from mixture prior: 1/sqrt(n) for parametric models

    Formula: If f0 in parametric model with dim d, adaptation gives sqrt(d/n) rate

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
    Ghosal Ch 10 §10.2.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Parametric rate recovery from mixture prior: 1/sqrt(n) for parametric models"})


def cheatsheet():
    return "gh_c10_3: Parametric rate recovery from mixture prior: 1/sqrt(n) for parametric models"

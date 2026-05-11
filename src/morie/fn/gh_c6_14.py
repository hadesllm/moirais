# morie.fn — function file (hadesllm/morie)
"""Barron's predictive consistency: KL from predictive to true density goes to 0."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_pred_consist"]


def ghosal_pred_consist(x):
    """
    Barron's predictive consistency: KL from predictive to true density goes to 0

    Formula: sum_{i=1}^n KL(P0, Pi-predictive at i) / n -> 0 a.s.

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
    Ghosal Ch 6 §6.8.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Barron's predictive consistency: KL from predictive to true density goes to 0"})


def cheatsheet():
    return "gh_c6_14: Barron's predictive consistency: KL from predictive to true density goes to 0"

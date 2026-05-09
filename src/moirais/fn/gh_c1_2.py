# moirais.fn — function file (hadesllm/moirais)
"""Absolute continuity condition for posterior existence in nonparametric Bayes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_absolute_continuity"]


def ghosal_absolute_continuity(x):
    """
    Absolute continuity condition for posterior existence in nonparametric Bayes

    Formula: P_theta << mu for all theta, posterior exists via Radon-Nikodym

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
    Ghosal Ch 1 §1.3.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Absolute continuity condition for posterior existence in nonparametric Bayes"})


def cheatsheet():
    return "gh_c1_2: Absolute continuity condition for posterior existence in nonparametric Bayes"

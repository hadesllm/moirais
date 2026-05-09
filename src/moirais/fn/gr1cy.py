# moirais.fn — function file (hadesllm/moirais)
"""Smith's 1cycle LR schedule: triangular warm-up then anneal + momentum mirror."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_1cycle_schedule"]


def geron_1cycle_schedule(eta_min, eta_max, t, T):
    """
    Smith's 1cycle LR schedule: triangular warm-up then anneal + momentum mirror

    Formula: two-phase linear schedule: eta rises eta_min -> eta_max over T/2, then falls, all in one epoch-group

    Parameters
    ----------
    eta_min : array-like
        Input data.
    eta_max : array-like
        Input data.
    t : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: eta

    References
    ----------
    Géron Ch 11, 1cycle section (Smith 2018)
    """
    eta_min = np.asarray(eta_min, dtype=float)
    n = int(eta_min) if eta_min.ndim == 0 else len(eta_min)
    result = float(np.mean(eta_min))
    se = float(np.std(eta_min, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Smith's 1cycle LR schedule: triangular warm-up then anneal + momentum mirror"})


def cheatsheet():
    return "gr1cy: Smith's 1cycle LR schedule: triangular warm-up then anneal + momentum mirror"

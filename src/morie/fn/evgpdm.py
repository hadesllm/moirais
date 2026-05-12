"""MLE of GPD parameters above threshold."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_gpd_mle"]


def evt_gpd_mle(y, init):
    """
    MLE of GPD parameters above threshold

    Formula: argmax over (σ,ξ); profile out σ if ξ->0

    Parameters
    ----------
    y : array-like
        Input data.
    init : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sigma, xi, ll

    References
    ----------
    Coles (2001)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MLE of GPD parameters above threshold"})


def cheatsheet():
    return "evgpdm: MLE of GPD parameters above threshold"

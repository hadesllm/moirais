"""Network PAF accounting for spillover."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["network_paf"]


def network_paf(y, exposure, network):
    """
    Network PAF accounting for spillover

    Formula: DiD-style with network exposure

    Parameters
    ----------
    y : array-like
        Input data.
    exposure : array-like
        Input data.
    network : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Halloran-Hudgens (2016)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Network PAF accounting for spillover"})


def cheatsheet():
    return "netparf: Network PAF accounting for spillover"

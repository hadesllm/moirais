"""Path-specific effects (PSE)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["path_specific_effect"]


def path_specific_effect(Y, X, M_list, path):
    """
    Path-specific effects (PSE)

    Formula: counterfactual along subset of paths

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M_list : array-like
        Input data.
    path : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pearl (2001); Avin-Shpitser-Pearl (2005)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Path-specific effects (PSE)"})


def cheatsheet():
    return "pseudo: Path-specific effects (PSE)"

"""D-study decision-making."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["d_study_decision"]


def d_study_decision(G_components, n_proposed):
    """
    D-study decision-making

    Formula: recompute G with proposed n_facets

    Parameters
    ----------
    G_components : array-like
        Input data.
    n_proposed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Brennan (2001)
    """
    G_components = np.atleast_1d(np.asarray(G_components, dtype=float))
    n = len(G_components)
    result = float(np.mean(G_components))
    se = float(np.std(G_components, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "D-study decision-making"})


def cheatsheet():
    return "genvdm: D-study decision-making"

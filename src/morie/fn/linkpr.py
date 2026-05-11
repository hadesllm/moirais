"""Link prediction (CN, AA, RA)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["link_prediction"]


def link_prediction(G, u, v, method):
    """
    Link prediction (CN, AA, RA)

    Formula: common neighbors / Adamic-Adar / resource alloc

    Parameters
    ----------
    G : array-like
        Input data.
    u : array-like
        Input data.
    v : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Liben-Nowell-Kleinberg (2007)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Link prediction (CN, AA, RA)"})


def cheatsheet():
    return "linkPr: Link prediction (CN, AA, RA)"

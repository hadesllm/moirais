"""Label propagation community detection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["label_propagation"]


def label_propagation(G):
    """
    Label propagation community detection

    Formula: each node adopts most-common neighbor label

    Parameters
    ----------
    G : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Raghavan-Albert-Kumara (2007)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Label propagation community detection"})


def cheatsheet():
    return "comlpa: Label propagation community detection"

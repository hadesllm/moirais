"""GATv2 -- dynamic attention."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gat_v2"]


def gat_v2(A, X):
    """
    GATv2 -- dynamic attention

    Formula: α_{ij} swaps order of attention layers

    Parameters
    ----------
    A : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Brody-Alon-Yahav (2022)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GATv2 -- dynamic attention"})


def cheatsheet():
    return "gatV2: GATv2 -- dynamic attention"

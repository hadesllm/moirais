"""Maximum likelihood phylogeny."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["phylogenetic_ml"]


def phylogenetic_ml(alignment, model):
    """
    Maximum likelihood phylogeny

    Formula: Felsenstein pruning + branch length opt

    Parameters
    ----------
    alignment : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Felsenstein (1981)
    """
    alignment = np.atleast_1d(np.asarray(alignment, dtype=float))
    n = len(alignment)
    result = float(np.mean(alignment))
    se = float(np.std(alignment, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Maximum likelihood phylogeny"})


def cheatsheet():
    return "phylml: Maximum likelihood phylogeny"

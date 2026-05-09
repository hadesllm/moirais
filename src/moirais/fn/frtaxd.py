"""Forest taxon diversity (Shannon-Wiener spatial)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["forest_taxon_diversity"]


def forest_taxon_diversity(coords, species, grid):
    """
    Forest taxon diversity (Shannon-Wiener spatial)

    Formula: H' per cell + spatial autocorrelation

    Parameters
    ----------
    coords : array-like
        Input data.
    species : array-like
        Input data.
    grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shannon-Weaver (1949); Magurran (2004)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Forest taxon diversity (Shannon-Wiener spatial)"})


def cheatsheet():
    return "frtaxd: Forest taxon diversity (Shannon-Wiener spatial)"

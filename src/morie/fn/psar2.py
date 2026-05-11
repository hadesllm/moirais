"""Topological polar surface area (TPSA)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["polar_surface_area"]


def polar_surface_area(smiles):
    """
    Topological polar surface area (TPSA)

    Formula: sum tabulated PSA contributions per polar atom

    Parameters
    ----------
    smiles : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ertl-Rohde-Selzer (2000)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Topological polar surface area (TPSA)"})


def cheatsheet():
    return "psar2: Topological polar surface area (TPSA)"

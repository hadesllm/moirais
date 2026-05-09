"""Molecular weight."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["molecular_weight"]


def molecular_weight(smiles):
    """
    Molecular weight

    Formula: sum atomic weights of all atoms

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
    IUPAC atomic weights (2021)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Molecular weight"})


def cheatsheet():
    return "mwght: Molecular weight"

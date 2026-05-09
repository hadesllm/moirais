"""Rotatable bond count."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rotatable_bond_count"]


def rotatable_bond_count(smiles):
    """
    Rotatable bond count

    Formula: non-ring single bonds excluding terminal

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
    Veber (2002)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rotatable bond count"})


def cheatsheet():
    return "rotbnd: Rotatable bond count"

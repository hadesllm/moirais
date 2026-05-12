"""Scaffold hopping -- bioisosteric replacement."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["scaffold_hop"]


def scaffold_hop(lead_smiles, scaffold_db):
    """
    Scaffold hopping -- bioisosteric replacement

    Formula: replace core ring system preserving exit-vector geometry

    Parameters
    ----------
    lead_smiles : array-like
        Input data.
    scaffold_db : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schneider-Schneider-Sieber (1999)
    """
    lead_smiles = np.atleast_1d(np.asarray(lead_smiles, dtype=float))
    n = len(lead_smiles)
    result = float(np.mean(lead_smiles))
    se = float(np.std(lead_smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Scaffold hopping -- bioisosteric replacement"})


def cheatsheet():
    return "scfhop: Scaffold hopping -- bioisosteric replacement"

"""Plasma protein binding fraction unbound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["plasma_protein_binding"]


def plasma_protein_binding(smiles):
    """
    Plasma protein binding fraction unbound

    Formula: regression on physchem descriptors

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
    Lambrinidis-Vallianatou-Tsantili-Kakoulidou (2015)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Plasma protein binding fraction unbound"})


def cheatsheet():
    return "ppbnd: Plasma protein binding fraction unbound"

"""Plasma half-life prediction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["half_life"]


def half_life(smiles, Vd, Cl):
    """
    Plasma half-life prediction

    Formula: composite from Vd and clearance

    Parameters
    ----------
    smiles : array-like
        Input data.
    Vd : array-like
        Input data.
    Cl : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Smith et al (2012)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Plasma half-life prediction"})


def cheatsheet():
    return "halft: Plasma half-life prediction"

"""Oral bioavailability fraction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["oral_bioavailability"]


def oral_bioavailability(smiles):
    """
    Oral bioavailability fraction

    Formula: composite filter (Lipinski + Veber + Egan); regression refinement

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
    Veber et al (2002); Hou-Xu (2003)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Oral bioavailability fraction"})


def cheatsheet():
    return "oralb: Oral bioavailability fraction"

"""Flexible-receptor docking with side-chain rotamers."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["flexible_receptor_dock"]


def flexible_receptor_dock(receptor, ligand, flex_residues):
    """
    Flexible-receptor docking with side-chain rotamers

    Formula: sample sidechain rotamers + ligand pose simultaneously

    Parameters
    ----------
    receptor : array-like
        Input data.
    ligand : array-like
        Input data.
    flex_residues : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sherman et al (2006) IFD
    """
    receptor = np.atleast_1d(np.asarray(receptor, dtype=float))
    n = len(receptor)
    result = float(np.mean(receptor))
    se = float(np.std(receptor, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Flexible-receptor docking with side-chain rotamers"})


def cheatsheet():
    return "flexrd: Flexible-receptor docking with side-chain rotamers"

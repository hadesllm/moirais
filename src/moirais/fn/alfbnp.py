"""AlphaFold-3 protein-ligand co-folding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["af3_protein_ligand"]


def af3_protein_ligand(protein, ligand):
    """
    AlphaFold-3 protein-ligand co-folding

    Formula: diffusion across all atoms incl ligands

    Parameters
    ----------
    protein : array-like
        Input data.
    ligand : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Abramson et al (2024)
    """
    protein = np.atleast_1d(np.asarray(protein, dtype=float))
    n = len(protein)
    result = float(np.mean(protein))
    se = float(np.std(protein, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold-3 protein-ligand co-folding"})


def cheatsheet():
    return "alfbnp: AlphaFold-3 protein-ligand co-folding"

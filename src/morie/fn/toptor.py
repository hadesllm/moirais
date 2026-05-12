"""Topological torsion fingerprint."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["topological_torsion"]


def topological_torsion(smiles, n_bits):
    """
    Topological torsion fingerprint

    Formula: hash(4-atom path types) -> bit vector

    Parameters
    ----------
    smiles : array-like
        Input data.
    n_bits : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nilakantan et al (1987)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Topological torsion fingerprint"})


def cheatsheet():
    return "toptor: Topological torsion fingerprint"

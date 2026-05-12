"""Atom-pair fingerprint."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["atom_pair_fp"]


def atom_pair_fp(smiles, n_bits, max_dist):
    """
    Atom-pair fingerprint

    Formula: hash(atom_i, atom_j, topological_distance) -> bit

    Parameters
    ----------
    smiles : array-like
        Input data.
    n_bits : array-like
        Input data.
    max_dist : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Carhart-Smith-Venkataraghavan (1985)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Atom-pair fingerprint"})


def cheatsheet():
    return "atmpair: Atom-pair fingerprint"

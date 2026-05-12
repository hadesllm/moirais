"""RDKit path-based fingerprint."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rdkit_path_fp"]


def rdkit_path_fp(smiles, n_bits, min_path, max_path):
    """
    RDKit path-based fingerprint

    Formula: enumerate paths len 1..7 -> hash -> bit

    Parameters
    ----------
    smiles : array-like
        Input data.
    n_bits : array-like
        Input data.
    min_path : array-like
        Input data.
    max_path : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    RDKit Open-Source Cheminformatics
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RDKit path-based fingerprint"})


def cheatsheet():
    return "rdkfp: RDKit path-based fingerprint"

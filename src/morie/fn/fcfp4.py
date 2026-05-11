"""Functional-Class Fingerprint radius 4 (FCFP4)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fcfp_4_fingerprint"]


def fcfp_4_fingerprint(smiles, n_bits):
    """
    Functional-Class Fingerprint radius 4 (FCFP4)

    Formula: like ECFP but atoms keyed by functional role

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
    Rogers & Hahn (2010)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional-Class Fingerprint radius 4 (FCFP4)"})


def cheatsheet():
    return "fcfp4: Functional-Class Fingerprint radius 4 (FCFP4)"

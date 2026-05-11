"""Avalon fingerprint."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["avalon_fingerprint"]


def avalon_fingerprint(smiles, n_bits):
    """
    Avalon fingerprint

    Formula: path-based 512/1024-bit

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
    Gedeck et al (2006) Merck
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Avalon fingerprint"})


def cheatsheet():
    return "avalon: Avalon fingerprint"

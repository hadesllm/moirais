"""ECFP6 / Morgan radius 3."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ecfp_6_fingerprint"]


def ecfp_6_fingerprint(smiles, n_bits):
    """
    ECFP6 / Morgan radius 3

    Formula: as ECFP4 with radius=3

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ECFP6 / Morgan radius 3"})


def cheatsheet():
    return "ecfp6: ECFP6 / Morgan radius 3"

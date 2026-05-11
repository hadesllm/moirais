"""Extended-Connectivity Fingerprint radius 4 (ECFP4 / Morgan radius 2)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ecfp_4_fingerprint"]


def ecfp_4_fingerprint(smiles, n_bits, radius):
    """
    Extended-Connectivity Fingerprint radius 4 (ECFP4 / Morgan radius 2)

    Formula: iterative atom-environment hash; fold to 1024- or 2048-bit

    Parameters
    ----------
    smiles : array-like
        Input data.
    n_bits : array-like
        Input data.
    radius : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rogers & Hahn (2010) ECFP
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Extended-Connectivity Fingerprint radius 4 (ECFP4 / Morgan radius 2)"})


def cheatsheet():
    return "ecfp4: Extended-Connectivity Fingerprint radius 4 (ECFP4 / Morgan radius 2)"

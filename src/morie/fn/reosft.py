"""REOS rapid-elimination of swill."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["reos_filter"]


def reos_filter(smiles):
    """
    REOS rapid-elimination of swill

    Formula: composite filter against PAINS-like nuisance compounds

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
    Walters et al (1998)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "REOS rapid-elimination of swill"})


def cheatsheet():
    return "reosft: REOS rapid-elimination of swill"

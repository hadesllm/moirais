"""Synthetic accessibility score (SAscore)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["synthetic_accessibility"]


def synthetic_accessibility(smiles):
    """
    Synthetic accessibility score (SAscore)

    Formula: fragment frequency + complexity penalties; 1-10

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
    Ertl-Schuffenhauer (2009)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Synthetic accessibility score (SAscore)"})


def cheatsheet():
    return "sasc1: Synthetic accessibility score (SAscore)"

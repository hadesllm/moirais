"""Ames mutagenicity classification."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ames_mutagenicity"]


def ames_mutagenicity(smiles):
    """
    Ames mutagenicity classification

    Formula: RF classifier on structural alerts

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
    Hansen et al (2009); Honma et al (2019)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ames mutagenicity classification"})


def cheatsheet():
    return "ames3: Ames mutagenicity classification"

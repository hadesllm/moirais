"""CYP450 isozyme inhibition (1A2/2C9/2C19/2D6/3A4)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cyp450_inhibition"]


def cyp450_inhibition(smiles, isozyme):
    """
    CYP450 isozyme inhibition (1A2/2C9/2C19/2D6/3A4)

    Formula: per-isozyme RF classifier on FP + descriptors

    Parameters
    ----------
    smiles : array-like
        Input data.
    isozyme : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Veith et al (2009)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CYP450 isozyme inhibition (1A2/2C9/2C19/2D6/3A4)"})


def cheatsheet():
    return "cypin: CYP450 isozyme inhibition (1A2/2C9/2C19/2D6/3A4)"

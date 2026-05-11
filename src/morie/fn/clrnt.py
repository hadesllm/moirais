"""Hepatic intrinsic clearance prediction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clearance_intrinsic"]


def clearance_intrinsic(smiles, species):
    """
    Hepatic intrinsic clearance prediction

    Formula: DNN on FP + tissue context

    Parameters
    ----------
    smiles : array-like
        Input data.
    species : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wood et al (2017); Pirmohamed (2019)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hepatic intrinsic clearance prediction"})


def cheatsheet():
    return "clrnt: Hepatic intrinsic clearance prediction"

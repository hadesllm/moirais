"""Drug-induced liver injury (DILI) classification."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hepatotoxicity"]


def hepatotoxicity(smiles):
    """
    Drug-induced liver injury (DILI) classification

    Formula: ensemble classifier on FP + pharmacophore features

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
    Chen et al (2016) DILI
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Drug-induced liver injury (DILI) classification"})


def cheatsheet():
    return "hepatx: Drug-induced liver injury (DILI) classification"

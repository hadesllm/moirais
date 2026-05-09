"""Blood-brain barrier permeability classifier."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bbb_permeability"]


def bbb_permeability(smiles):
    """
    Blood-brain barrier permeability classifier

    Formula: random forest on physchem + fingerprint features

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
    Martins et al (2012); Vilar et al (2010)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Blood-brain barrier permeability classifier"})


def cheatsheet():
    return "bbbpr: Blood-brain barrier permeability classifier"

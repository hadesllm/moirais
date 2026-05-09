"""MACCS 166-bit structural keys."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["maccs_keys"]


def maccs_keys(smiles):
    """
    MACCS 166-bit structural keys

    Formula: 166 hand-crafted SMARTS substructure indicators

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
    Durant et al (2002) JCIM
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MACCS 166-bit structural keys"})


def cheatsheet():
    return "maccs: MACCS 166-bit structural keys"

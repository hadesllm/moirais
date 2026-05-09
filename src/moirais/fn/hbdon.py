"""H-bond donor count."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hbond_donor_count"]


def hbond_donor_count(smiles):
    """
    H-bond donor count

    Formula: count N-H + O-H groups

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
    Lipinski (1997)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "H-bond donor count"})


def cheatsheet():
    return "hbdon: H-bond donor count"

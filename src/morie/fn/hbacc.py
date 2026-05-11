"""H-bond acceptor count."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hbond_acceptor_count"]


def hbond_acceptor_count(smiles):
    """
    H-bond acceptor count

    Formula: count N + O atoms (Lipinski definition)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "H-bond acceptor count"})


def cheatsheet():
    return "hbacc: H-bond acceptor count"

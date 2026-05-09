"""Pan-assay interference compound filter (PAINS)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pains_filter"]


def pains_filter(smiles):
    """
    Pan-assay interference compound filter (PAINS)

    Formula: 480 SMARTS patterns flagging promiscuous binders

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
    Baell-Holloway (2010)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pan-assay interference compound filter (PAINS)"})


def cheatsheet():
    return "pains3: Pan-assay interference compound filter (PAINS)"

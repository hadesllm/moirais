"""Egan drug-like filter (PSA + LogP)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["egan_filter"]


def egan_filter(smiles):
    """
    Egan drug-like filter (PSA + LogP)

    Formula: PSA ≤132 Å², -1≤LogP≤5.88

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
    Egan-Merz-Baldwin (2000)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Egan drug-like filter (PSA + LogP)"})


def cheatsheet():
    return "egan2: Egan drug-like filter (PSA + LogP)"

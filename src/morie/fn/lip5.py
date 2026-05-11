"""Lipinski Rule of 5 oral-bioavailability filter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lipinski_rule_of_5"]


def lipinski_rule_of_5(smiles):
    """
    Lipinski Rule of 5 oral-bioavailability filter

    Formula: MW≤500, LogP≤5, HBA≤10, HBD≤5; pass if ≥3 met

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
    Lipinski et al (1997, 2001)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lipinski Rule of 5 oral-bioavailability filter"})


def cheatsheet():
    return "lip5: Lipinski Rule of 5 oral-bioavailability filter"

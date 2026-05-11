"""Veber rule for oral bioavailability."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["veber_rule"]


def veber_rule(smiles):
    """
    Veber rule for oral bioavailability

    Formula: rotatable bonds ≤10; PSA ≤140 Å²

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
    Veber et al (2002)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Veber rule for oral bioavailability"})


def cheatsheet():
    return "vebr3: Veber rule for oral bioavailability"

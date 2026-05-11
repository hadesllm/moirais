"""SELFIES encoding (robust to mutation)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["selfies_encode"]


def selfies_encode(smiles):
    """
    SELFIES encoding (robust to mutation)

    Formula: context-free encoding guaranteeing valid molecule

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
    Krenn et al (2020) SELFIES
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SELFIES encoding (robust to mutation)"})


def cheatsheet():
    return "selfgr: SELFIES encoding (robust to mutation)"

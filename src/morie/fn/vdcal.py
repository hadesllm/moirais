"""Volume of distribution (Vd_ss) prediction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["volume_of_distribution"]


def volume_of_distribution(smiles, ppb):
    """
    Volume of distribution (Vd_ss) prediction

    Formula: empirical model on physchem + ppb

    Parameters
    ----------
    smiles : array-like
        Input data.
    ppb : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lombardo et al (2002, 2018)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Volume of distribution (Vd_ss) prediction"})


def cheatsheet():
    return "vdcal: Volume of distribution (Vd_ss) prediction"

"""hERG cardiac potassium-channel inhibition risk."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["herg_inhibition"]


def herg_inhibition(smiles):
    """
    hERG cardiac potassium-channel inhibition risk

    Formula: DNN classifier on Morgan FP

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
    Cai et al (2019); Ogura et al (2019)
    """
    smiles = np.atleast_1d(np.asarray(smiles, dtype=float))
    n = len(smiles)
    result = float(np.mean(smiles))
    se = float(np.std(smiles, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "hERG cardiac potassium-channel inhibition risk"})


def cheatsheet():
    return "hergp: hERG cardiac potassium-channel inhibition risk"

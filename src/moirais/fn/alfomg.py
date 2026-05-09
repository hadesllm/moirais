"""OpenFold MSA-pair head."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["openfold_msa_pair"]


def openfold_msa_pair(msa, pair):
    """
    OpenFold MSA-pair head

    Formula: open-source AlphaFold reproduction

    Parameters
    ----------
    msa : array-like
        Input data.
    pair : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ahdritz et al (2022)
    """
    msa = np.atleast_1d(np.asarray(msa, dtype=float))
    n = len(msa)
    result = float(np.mean(msa))
    se = float(np.std(msa, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "OpenFold MSA-pair head"})


def cheatsheet():
    return "alfomg: OpenFold MSA-pair head"

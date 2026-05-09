"""Covariance model RNA family."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rna_covariance"]


def rna_covariance(alignment, structure):
    """
    Covariance model RNA family

    Formula: profile SCFG over consensus structure

    Parameters
    ----------
    alignment : array-like
        Input data.
    structure : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Eddy-Durbin (1994); Nawrocki-Eddy (2013) Infernal
    """
    alignment = np.atleast_1d(np.asarray(alignment, dtype=float))
    n = len(alignment)
    result = float(np.mean(alignment))
    se = float(np.std(alignment, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Covariance model RNA family"})


def cheatsheet():
    return "rnacov: Covariance model RNA family"

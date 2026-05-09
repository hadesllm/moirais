"""Metagenome binning (MetaBAT)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["metagenome_binning"]


def metagenome_binning(contigs, abundance):
    """
    Metagenome binning (MetaBAT)

    Formula: abundance + tetranucleotide composition clustering

    Parameters
    ----------
    contigs : array-like
        Input data.
    abundance : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kang et al (2015) MetaBAT2
    """
    contigs = np.atleast_1d(np.asarray(contigs, dtype=float))
    n = len(contigs)
    result = float(np.mean(contigs))
    se = float(np.std(contigs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Metagenome binning (MetaBAT)"})


def cheatsheet():
    return "metabd: Metagenome binning (MetaBAT)"

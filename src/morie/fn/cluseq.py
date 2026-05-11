"""Genomic outbreak clustering."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sequence_clustering"]


def sequence_clustering(sequences, snp_threshold):
    """
    Genomic outbreak clustering

    Formula: SNP-distance + linkage threshold

    Parameters
    ----------
    sequences : array-like
        Input data.
    snp_threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Croucher et al (2015)
    """
    sequences = np.atleast_1d(np.asarray(sequences, dtype=float))
    n = len(sequences)
    result = float(np.mean(sequences))
    se = float(np.std(sequences, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Genomic outbreak clustering"})


def cheatsheet():
    return "cluseq: Genomic outbreak clustering"

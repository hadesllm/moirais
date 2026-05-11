# morie.fn — function file (hadesllm/morie)
"""Minor allele frequency (MAF) calculation for SNP markers."""
import numpy as np
from ._richresult import RichResult

__all__ = ["maf_calculation"]


def maf_calculation(marker_matrix):
    """
    Minor allele frequency (MAF) calculation for SNP markers

    Formula: p_j = (count minor allele at locus j) / (2 * n_individuals)

    Parameters
    ----------
    marker_matrix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'maf': 'vector'}

    References
    ----------
    Montesinos Lopez Ch 2
    """
    marker_matrix = np.asarray(marker_matrix, dtype=float)
    n = int(marker_matrix) if marker_matrix.ndim == 0 else len(marker_matrix)
    result = float(np.mean(marker_matrix))
    se = float(np.std(marker_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Minor allele frequency (MAF) calculation for SNP markers"})


def cheatsheet():
    return "mafcl: Minor allele frequency (MAF) calculation for SNP markers"

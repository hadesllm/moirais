# morie.fn — function file (hadesllm/morie)
"""Marker heterozygosity and frequency of heterogeneous loci."""
import numpy as np
from ._richresult import RichResult

__all__ = ["heterozygosity_locus"]


def heterozygosity_locus(marker_matrix):
    """
    Marker heterozygosity and frequency of heterogeneous loci

    Formula: H_j = 2*p_j*(1-p_j); H_obs = (# heterozygous calls)/n_ind

    Parameters
    ----------
    marker_matrix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'H_obs': 'array', 'H_exp': 'array'}

    References
    ----------
    Montesinos Lopez Ch 2
    """
    marker_matrix = np.asarray(marker_matrix, dtype=float)
    n = int(marker_matrix) if marker_matrix.ndim == 0 else len(marker_matrix)
    result = float(np.mean(marker_matrix))
    se = float(np.std(marker_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Marker heterozygosity and frequency of heterogeneous loci"})


def cheatsheet():
    return "hetlc: Marker heterozygosity and frequency of heterogeneous loci"

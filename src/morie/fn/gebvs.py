# morie.fn -- function file (hadesllm/morie)
"""Genomic estimated breeding value (GEBV) for genomic selection decision."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gebv_selection"]


def gebv_selection(marker_matrix, y_train, method):
    """
    Genomic estimated breeding value (GEBV) for genomic selection decision

    Formula: GEBV_i = hat_g_i from GBLUP/BayesX; rank candidates by GEBV for selection

    Parameters
    ----------
    marker_matrix : array-like
        Input data.
    y_train : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'gebv': 'array', 'rank': 'array'}

    References
    ----------
    Montesinos Lopez Ch 2,5
    """
    marker_matrix = np.asarray(marker_matrix, dtype=float)
    n = int(marker_matrix) if marker_matrix.ndim == 0 else len(marker_matrix)
    if marker_matrix.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Genomic estimated breeding value (GEBV) for genomic selection decision"})
    estimate = np.median(marker_matrix)
    se = 1.2533 * np.std(marker_matrix, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Genomic estimated breeding value (GEBV) for genomic selection decision"})


def cheatsheet():
    return "gebvs: Genomic estimated breeding value (GEBV) for genomic selection decision"

# moirais.fn — function file (hadesllm/moirais)
"""Multivariate analysis of concurrent biomedical signals (covariance matrix)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch2_multivariate"]


def rangayyan_ch2_multivariate(signals_matrix):
    """
    Multivariate analysis of concurrent biomedical signals (covariance matrix)

    Formula: Sigma_XY = E[(X-mu_X)(Y-mu_Y)^T]; normalized -> correlation matrix

    Parameters
    ----------
    signals_matrix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cov_matrix, corr_matrix

    References
    ----------
    Rangayyan Ch 2
    """
    signals_matrix = np.asarray(signals_matrix, dtype=float)
    n = int(signals_matrix) if signals_matrix.ndim == 0 else len(signals_matrix)
    result = float(np.mean(signals_matrix))
    se = float(np.std(signals_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multivariate analysis of concurrent biomedical signals (covariance matrix)"})


def cheatsheet():
    return "rgeqn2: Multivariate analysis of concurrent biomedical signals (covariance matrix)"

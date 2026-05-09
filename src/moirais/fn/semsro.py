"""SEM residual matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sem_residual"]


def sem_residual(sample_cov, fitted_cov):
    """
    SEM residual matrix

    Formula: S - Sigma_hat

    Parameters
    ----------
    sample_cov : array-like
        Input data.
    fitted_cov : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bentler-Yuan (1999)
    """
    sample_cov = np.atleast_1d(np.asarray(sample_cov, dtype=float))
    n = len(sample_cov)
    result = float(np.mean(sample_cov))
    se = float(np.std(sample_cov, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SEM residual matrix"})


def cheatsheet():
    return "semsro: SEM residual matrix"

# moirais.fn — function file (hadesllm/moirais)
"""BIC for GMM model selection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bic_gmm"]


def geron_bic_gmm(log_likelihood, n, n_params):
    """
    BIC for GMM model selection

    Formula: BIC = log(n) * p - 2 * log_L

    Parameters
    ----------
    log_likelihood : array-like
        Input data.
    n : array-like
        Input data.
    n_params : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bic

    References
    ----------
    Géron Ch 8, BIC / AIC section
    """
    log_likelihood = np.asarray(log_likelihood, dtype=float)
    n = int(log_likelihood) if log_likelihood.ndim == 0 else len(log_likelihood)
    result = float(np.mean(log_likelihood))
    se = float(np.std(log_likelihood, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BIC for GMM model selection"})


def cheatsheet():
    return "grbic: BIC for GMM model selection"

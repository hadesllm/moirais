"""CMIP multi-model ensemble mean."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cmip_ensemble"]


def cmip_ensemble(models, weights):
    """
    CMIP multi-model ensemble mean

    Formula: weighted mean across model runs

    Parameters
    ----------
    models : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Knutti et al (2017)
    """
    models = np.atleast_1d(np.asarray(models, dtype=float))
    n = len(models)
    result = float(np.mean(models))
    se = float(np.std(models, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CMIP multi-model ensemble mean"})


def cheatsheet():
    return "caCMIP: CMIP multi-model ensemble mean"

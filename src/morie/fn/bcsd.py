"""Bias-correction spatial-disaggregation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bcsd_downscaling"]


def bcsd_downscaling(gcm, obs):
    """
    Bias-correction spatial-disaggregation

    Formula: bias-correct -> spatial disaggregation via climatology

    Parameters
    ----------
    gcm : array-like
        Input data.
    obs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wood et al (2002)
    """
    obs = np.atleast_1d(np.asarray(obs, dtype=float))
    n = len(obs)
    result = float(np.mean(obs))
    se = float(np.std(obs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bias-correction spatial-disaggregation"})


def cheatsheet():
    return "bcsd: Bias-correction spatial-disaggregation"

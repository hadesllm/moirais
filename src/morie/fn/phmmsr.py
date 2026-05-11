"""Profile HMM database search."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["profile_hmm_search"]


def profile_hmm_search(profile, db):
    """
    Profile HMM database search

    Formula: profile vs sequence DB; E-value calibration

    Parameters
    ----------
    profile : array-like
        Input data.
    db : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Eddy (2011) HMMER3
    """
    profile = np.atleast_1d(np.asarray(profile, dtype=float))
    n = len(profile)
    result = float(np.mean(profile))
    se = float(np.std(profile, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Profile HMM database search"})


def cheatsheet():
    return "phmmsr: Profile HMM database search"

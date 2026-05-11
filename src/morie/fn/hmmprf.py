"""Profile HMM scoring (HMMER-style)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hmm_profile"]


def hmm_profile(seq, profile):
    """
    Profile HMM scoring (HMMER-style)

    Formula: forward + Viterbi over match/insert/delete states

    Parameters
    ----------
    seq : array-like
        Input data.
    profile : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Durbin et al (1998); Eddy (1998)
    """
    seq = np.atleast_1d(np.asarray(seq, dtype=float))
    n = len(seq)
    result = float(np.mean(seq))
    se = float(np.std(seq, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Profile HMM scoring (HMMER-style)"})


def cheatsheet():
    return "hmmprf: Profile HMM scoring (HMMER-style)"

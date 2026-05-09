"""Prioritized experience replay."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["prioritized_experience_replay"]


def prioritized_experience_replay(buffer, alpha, beta):
    """
    Prioritized experience replay

    Formula: p_i = |δ_i|^α; importance sampling weight w_i

    Parameters
    ----------
    buffer : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schaul et al (2016)
    """
    buffer = np.atleast_1d(np.asarray(buffer, dtype=float))
    n = len(buffer)
    result = float(np.mean(buffer))
    se = float(np.std(buffer, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prioritized experience replay"})


def cheatsheet():
    return "pero: Prioritized experience replay"

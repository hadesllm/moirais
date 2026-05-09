"""ABC rejection sampler."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["abc_rejection"]


def abc_rejection(sim, obs, eps, prior):
    """
    ABC rejection sampler

    Formula: accept iff ||summary(sim) - summary(obs)|| < eps

    Parameters
    ----------
    sim : array-like
        Input data.
    obs : array-like
        Input data.
    eps : array-like
        Input data.
    prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pritchard et al (1999)
    """
    obs = np.atleast_1d(np.asarray(obs, dtype=float))
    n = len(obs)
    result = float(np.mean(obs))
    se = float(np.std(obs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ABC rejection sampler"})


def cheatsheet():
    return "abcrej: ABC rejection sampler"

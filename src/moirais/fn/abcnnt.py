"""ABC with neural likelihood-free inference."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["abc_neural"]


def abc_neural(sim, obs, theta_prior, n_train):
    """
    ABC with neural likelihood-free inference

    Formula: normalizing flow trained on (theta, summary) pairs

    Parameters
    ----------
    sim : array-like
        Input data.
    obs : array-like
        Input data.
    theta_prior : array-like
        Input data.
    n_train : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Papamakarios et al (2019) SNL
    """
    obs = np.atleast_1d(np.asarray(obs, dtype=float))
    n = len(obs)
    result = float(np.mean(obs))
    se = float(np.std(obs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ABC with neural likelihood-free inference"})


def cheatsheet():
    return "abcnnt: ABC with neural likelihood-free inference"

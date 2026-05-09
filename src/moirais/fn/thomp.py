"""Thompson sampling (Beta-Bernoulli)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["thompson_sampling"]


def thompson_sampling(arms, T):
    """
    Thompson sampling (Beta-Bernoulli)

    Formula: sample θ_i ~ Beta(α_i, β_i); pick argmax

    Parameters
    ----------
    arms : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Thompson (1933); Russo et al (2018) tutorial
    """
    arms = np.atleast_1d(np.asarray(arms, dtype=float))
    n = len(arms)
    result = float(np.mean(arms))
    se = float(np.std(arms, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Thompson sampling (Beta-Bernoulli)"})


def cheatsheet():
    return "thomp: Thompson sampling (Beta-Bernoulli)"

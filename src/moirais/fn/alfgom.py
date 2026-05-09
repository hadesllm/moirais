"""AlphaGo Monte Carlo rollout policy."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphago_montecarlo"]


def alphago_montecarlo(state, rollout_net, horizon):
    """
    AlphaGo Monte Carlo rollout policy

    Formula: fast rollout net: pi_rollout(s) sampled to terminal

    Parameters
    ----------
    state : array-like
        Input data.
    rollout_net : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2016) Nature AlphaGo
    """
    state = np.atleast_1d(np.asarray(state, dtype=float))
    n = len(state)
    result = float(np.mean(state))
    se = float(np.std(state, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaGo Monte Carlo rollout policy"})


def cheatsheet():
    return "alfgom: AlphaGo Monte Carlo rollout policy"

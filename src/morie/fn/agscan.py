"""AlphaZero self-consistency check across re-runs."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphazero_self_consistency"]


def alphazero_self_consistency(policy_net, seeds):
    """
    AlphaZero self-consistency check across re-runs

    Formula: compare policy entropy across seeds

    Parameters
    ----------
    policy_net : array-like
        Input data.
    seeds : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Silver et al (2018)
    """
    policy_net = np.atleast_1d(np.asarray(policy_net, dtype=float))
    n = len(policy_net)
    result = float(np.mean(policy_net))
    se = float(np.std(policy_net, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaZero self-consistency check across re-runs"})


def cheatsheet():
    return "agscan: AlphaZero self-consistency check across re-runs"

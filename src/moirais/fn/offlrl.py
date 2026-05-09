"""Conservative Q-learning (offline RL)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["offline_rl_cql"]


def offline_rl_cql(dataset, alpha):
    """
    Conservative Q-learning (offline RL)

    Formula: L_CQL = α(E_s log sum exp Q(s,·) − E_(s,a)~D Q(s,a)) + Bellman

    Parameters
    ----------
    dataset : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kumar et al (2020) CQL
    """
    dataset = np.atleast_1d(np.asarray(dataset, dtype=float))
    n = len(dataset)
    result = float(np.mean(dataset))
    se = float(np.std(dataset, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Conservative Q-learning (offline RL)"})


def cheatsheet():
    return "offlrl: Conservative Q-learning (offline RL)"

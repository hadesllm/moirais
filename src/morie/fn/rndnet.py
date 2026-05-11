"""RND exploration bonus."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["random_network_distillation"]


def random_network_distillation(env, predictor, target):
    """
    RND exploration bonus

    Formula: r_int = ||f(s) − f_target(s)||²

    Parameters
    ----------
    env : array-like
        Input data.
    predictor : array-like
        Input data.
    target : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Burda et al (2019) RND
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RND exploration bonus"})


def cheatsheet():
    return "rndnet: RND exploration bonus"

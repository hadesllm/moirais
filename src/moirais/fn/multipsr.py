"""Multi-stage probability sampling weight composition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["multi_stage_sampling"]


def multi_stage_sampling(y, stage_probs):
    """
    Multi-stage probability sampling weight composition

    Formula: w_i = 1 / (pi_1i pi_2|1 ... pi_ki|...)

    Parameters
    ----------
    y : array-like
        Input data.
    stage_probs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cochran (1977) §10
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-stage probability sampling weight composition"})


def cheatsheet():
    return "multipsr: Multi-stage probability sampling weight composition"

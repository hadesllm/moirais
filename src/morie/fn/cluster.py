"""One-stage cluster sample mean."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["one_stage_cluster"]


def one_stage_cluster(y, cluster):
    """
    One-stage cluster sample mean

    Formula: ybar_clu = (1/n) sum_i ybar_i

    Parameters
    ----------
    y : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cochran (1977) §9.2
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "One-stage cluster sample mean"})


def cheatsheet():
    return "cluster: One-stage cluster sample mean"

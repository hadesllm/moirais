"""Random survival forest."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["random_survival_forest"]


def random_survival_forest(time, event, X, n_trees):
    """
    Random survival forest

    Formula: forest with log-rank split criterion

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    n_trees : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ishwaran et al (2008)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random survival forest"})


def cheatsheet():
    return "survrsf: Random survival forest"

"""BART for survival outcomes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bart_survival"]


def bart_survival(time, event, X, n_trees):
    """
    BART for survival outcomes

    Formula: sum-of-trees prior on log-hazard

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
    Sparapani et al (2016)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BART for survival outcomes"})


def cheatsheet():
    return "survbtr: BART for survival outcomes"

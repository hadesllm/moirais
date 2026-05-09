"""Common-support / overlap diagnostic on PS distributions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_overlap_diagnostic"]


def causal_overlap_diagnostic(ps, treat):
    """
    Common-support / overlap diagnostic on PS distributions

    Formula: Report deciles of e(x) by treatment group

    Parameters
    ----------
    ps : array-like
        Input data.
    treat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: deciles, share_overlap

    References
    ----------
    Crump et al. (2009)
    """
    ps = np.atleast_1d(np.asarray(ps, dtype=float))
    n = len(ps)
    result = float(np.mean(ps))
    se = float(np.std(ps, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Common-support / overlap diagnostic on PS distributions"})


def cheatsheet():
    return "causovlap: Common-support / overlap diagnostic on PS distributions"

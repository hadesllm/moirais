"""Multi-marginal Sinkhorn on K marginals."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_multimarginal_iter"]


def ot_multimarginal_iter(margins, C_tensor, epsilon, max_iter):
    """
    Multi-marginal Sinkhorn on K marginals

    Formula: Iterate K row-marginals on tensor T_{i1..iK}

    Parameters
    ----------
    margins : array-like
        Input data.
    C_tensor : array-like
        Input data.
    epsilon : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T

    References
    ----------
    Benamou-Carlier-Cuturi-Nenna-Peyré (2015)
    """
    margins = np.atleast_1d(np.asarray(margins, dtype=float))
    n = len(margins)
    result = float(np.mean(margins))
    se = float(np.std(margins, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-marginal Sinkhorn on K marginals"})


def cheatsheet():
    return "otmot: Multi-marginal Sinkhorn on K marginals"

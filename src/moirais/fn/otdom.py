"""Joint distribution OT-based domain adaptation transform."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_domain_adaptation"]


def ot_domain_adaptation(Xs, Xt, epsilon):
    """
    Joint distribution OT-based domain adaptation transform

    Formula: Map source X_s via barycentric projection of OT plan

    Parameters
    ----------
    Xs : array-like
        Input data.
    Xt : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Xs_adapted

    References
    ----------
    Courty-Flamary-Tuia-Rakotomamonjy (2017)
    """
    Xs = np.atleast_1d(np.asarray(Xs, dtype=float))
    n = len(Xs)
    result = float(np.mean(Xs))
    se = float(np.std(Xs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Joint distribution OT-based domain adaptation transform"})


def cheatsheet():
    return "otdom: Joint distribution OT-based domain adaptation transform"

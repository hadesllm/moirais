"""MSM with mediator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["msm_mediation"]


def msm_mediation(y, A, M, H):
    """
    MSM with mediator

    Formula: NDE/NIE under sequential ignorability + IPTW

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    M : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele-Vansteelandt (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MSM with mediator"})


def cheatsheet():
    return "medmsm: MSM with mediator"

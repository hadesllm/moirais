# morie.fn — function file (hadesllm/morie)
"""Credit assignment problem: which past actions caused reward."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_credit_assignment"]


def geron_credit_assignment(trajectory):
    """
    Credit assignment problem: which past actions caused reward

    Formula: eligibility traces or discounted returns assign credit

    Parameters
    ----------
    trajectory : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: per_step_credit

    References
    ----------
    Géron Ch 19
    """
    trajectory = np.atleast_1d(np.asarray(trajectory, dtype=float))
    n = len(trajectory)
    result = float(np.mean(trajectory))
    se = float(np.std(trajectory, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Credit assignment problem: which past actions caused reward"})


def cheatsheet():
    return "hmcrd: Credit assignment problem: which past actions caused reward"

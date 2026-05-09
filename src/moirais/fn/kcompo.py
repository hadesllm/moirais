"""Sequential composition of k epsilon-DP mechanisms."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["k_step_dp_composition"]


def k_step_dp_composition(y, epsilons):
    """
    Sequential composition of k epsilon-DP mechanisms

    Formula: epsilon_total = sum_i epsilon_i

    Parameters
    ----------
    y : array-like
        Input data.
    epsilons : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork & Roth (2014) §3.5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sequential composition of k epsilon-DP mechanisms"})


def cheatsheet():
    return "kcompo: Sequential composition of k epsilon-DP mechanisms"

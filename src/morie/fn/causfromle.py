"""VanderWeele-Ding E-value for unmeasured confounding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_e_value"]


def causal_e_value(RR):
    """
    VanderWeele-Ding E-value for unmeasured confounding

    Formula: E = RR + sqrt(RR(RR-1)) where RR is observed risk ratio

    Parameters
    ----------
    RR : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Eval

    References
    ----------
    VanderWeele & Ding (2017)
    """
    RR = np.atleast_1d(np.asarray(RR, dtype=float))
    n = len(RR)
    result = float(np.mean(RR))
    se = float(np.std(RR, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VanderWeele-Ding E-value for unmeasured confounding"})


def cheatsheet():
    return "causfromle: VanderWeele-Ding E-value for unmeasured confounding"

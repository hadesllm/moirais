# morie.fn -- function file (hadesllm/morie)
"""ARE values for normal distribution: Wilcoxon=3/pi, sign=2/pi relative to t."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_are_normal_case"]


def gibbons_are_normal_case(distribution):
    """
    ARE values for normal distribution: Wilcoxon=3/pi, sign=2/pi relative to t

    Formula: ARE(Wilcoxon, t|normal) = 3/pi = 0.955; ARE(sign, t|normal) = 2/pi = 0.637

    Parameters
    ----------
    distribution : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ARE_values

    References
    ----------
    Gibbons Ch 13
    """
    distribution = np.asarray(distribution, dtype=float)
    n = int(distribution) if distribution.ndim == 0 else len(distribution)
    result = float(np.mean(distribution))
    se = float(np.std(distribution, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ARE values for normal distribution: Wilcoxon=3/pi, sign=2/pi relative to t"})


def cheatsheet():
    return "gb_are2: ARE values for normal distribution: Wilcoxon=3/pi, sign=2/pi relative to t"

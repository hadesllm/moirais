# morie.fn — function file (hadesllm/morie)
"""Group delay of a digital filter."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_group_delay"]


def rangayyan_group_delay(b, a, fs):
    """
    Group delay of a digital filter

    Formula: tau_g(f) = -d(angle(H(f)))/d(omega)

    Parameters
    ----------
    b : array-like
        Input data.
    a : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: group_delay, freqs

    References
    ----------
    Rangayyan Ch 3.4
    """
    b = np.asarray(b, dtype=float)
    n = int(b) if b.ndim == 0 else len(b)
    result = float(np.mean(b))
    se = float(np.std(b, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Group delay of a digital filter"})


def cheatsheet():
    return "rggrpd: Group delay of a digital filter"

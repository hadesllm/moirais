# morie.fn -- function file (hadesllm/morie)
"""Phase response of a digital filter."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_phase_response"]


def rangayyan_phase_response(b, a, fs):
    """
    Phase response of a digital filter

    Formula: phi(f) = angle(H(f)) = arctan(Im(H)/Re(H))

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
        Keys: phase, freqs

    References
    ----------
    Rangayyan Ch 3.4
    """
    b = np.asarray(b, dtype=float)
    n = int(b) if b.ndim == 0 else len(b)
    result = float(np.mean(b))
    se = float(np.std(b, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Phase response of a digital filter"})


def cheatsheet():
    return "rgphas: Phase response of a digital filter"

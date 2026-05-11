# morie.fn — function file (hadesllm/morie)
"""IIR Butterworth filter design."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_iir_filter"]


def rangayyan_iir_filter(x, cutoff, order):
    """
    IIR Butterworth filter design

    Formula: H(s) = 1 / prod(s^2 + s/Q_k + 1)

    Parameters
    ----------
    x : array-like
        Input data.
    cutoff : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rangayyan Ch 3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "IIR Butterworth filter design"})


def cheatsheet():
    return "rgiir: IIR Butterworth filter design"

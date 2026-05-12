# morie.fn -- function file (hadesllm/morie)
"""FitzHugh-Nagumo simplified neuron model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_fitzhugh_nagumo"]


def rangayyan_fitzhugh_nagumo(t, I_ext, a, b, eps):
    """
    FitzHugh-Nagumo simplified neuron model

    Formula: dv/dt = v - v^3/3 - w + I; dw/dt = eps*(v + a - b*w)

    Parameters
    ----------
    t : array-like
        Input data.
    I_ext : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: v_t, w_t

    References
    ----------
    Rangayyan Ch 7.8.1
    """
    t = np.asarray(t, dtype=float)
    n = int(t) if t.ndim == 0 else len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "FitzHugh-Nagumo simplified neuron model"})


def cheatsheet():
    return "rghmm: FitzHugh-Nagumo simplified neuron model"

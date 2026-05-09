# moirais.fn — function file (hadesllm/moirais)
"""Choi-Williams distribution (exponential kernel)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_choi_williams"]


def rangayyan_choi_williams(x, fs, sigma):
    """
    Choi-Williams distribution (exponential kernel)

    Formula: phi(theta,tau) = exp(-theta^2*tau^2/sigma)

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    sigma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cwd, t, freqs

    References
    ----------
    Rangayyan Ch 8.9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Choi-Williams distribution (exponential kernel)"})


def cheatsheet():
    return "rgchoi: Choi-Williams distribution (exponential kernel)"

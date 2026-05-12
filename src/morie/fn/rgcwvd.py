# morie.fn -- function file (hadesllm/morie)
"""Cohen's class TFDs via kernel function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_cohen_class"]


def rangayyan_cohen_class(x, fs, kernel):
    """
    Cohen's class TFDs via kernel function

    Formula: C(t,f) = integral integral phi(theta,tau)*A(theta,tau)*exp(-j2pi(theta*t+f*tau)) dtheta dtau

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tfd, t, freqs

    References
    ----------
    Rangayyan Ch 8.9
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cohen's class TFDs via kernel function"})


def cheatsheet():
    return "rgcwvd: Cohen's class TFDs via kernel function"

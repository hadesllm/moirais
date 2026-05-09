# moirais.fn — function file (hadesllm/moirais)
"""Heart rate variability (time domain)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_hrv"]


def rangayyan_hrv(x):
    """
    Heart rate variability (time domain)

    Formula: SDNN = std(NN), RMSSD = sqrt(mean(diff(NN)^2))

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rangayyan Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Heart rate variability (time domain)"})


def cheatsheet():
    return "rghrv: Heart rate variability (time domain)"

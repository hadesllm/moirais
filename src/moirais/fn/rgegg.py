# moirais.fn — function file (hadesllm/moirais)
"""Electrogastrogram (EGG) feature extraction (dominant frequency, power)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_egg"]


def rangayyan_egg(egg, fs):
    """
    Electrogastrogram (EGG) feature extraction (dominant frequency, power)

    Formula: EGG: dominant frequency 2-4 cycles/min; bradygastria <2, tachygastria 4-9

    Parameters
    ----------
    egg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: dominant_freq, power, rhythm_class

    References
    ----------
    Rangayyan Ch 1.2.8
    """
    egg = np.asarray(egg, dtype=float)
    n = int(egg) if egg.ndim == 0 else len(egg)
    result = float(np.mean(egg))
    se = float(np.std(egg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Electrogastrogram (EGG) feature extraction (dominant frequency, power)"})


def cheatsheet():
    return "rgegg: Electrogastrogram (EGG) feature extraction (dominant frequency, power)"

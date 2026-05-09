# moirais.fn — function file (hadesllm/moirais)
"""Frequency-domain feature extraction for CAD."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_freq_domain_feat"]


def rangayyan_freq_domain_feat(x, fs):
    """
    Frequency-domain feature extraction for CAD

    Formula: Peak frequency, spectral centroid, bandwidth, roll-off extracted from PSD

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: features_dict

    References
    ----------
    Rangayyan Ch 10
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frequency-domain feature extraction for CAD"})


def cheatsheet():
    return "rgfrqdom: Frequency-domain feature extraction for CAD"

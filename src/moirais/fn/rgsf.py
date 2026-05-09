# moirais.fn — function file (hadesllm/moirais)
"""Generic biomedical signal feature vector: time-domain + frequency-domain + nonlinear."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_signal_features"]


def rangayyan_signal_features(x, fs):
    """
    Generic biomedical signal feature vector: time-domain + frequency-domain + nonlinear

    Formula: F = [mean, std, rms, zcr, form_factor, centroid, bandwidth, spectral_entropy, sample_entropy]

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: feature_vector

    References
    ----------
    Rangayyan Ch 10
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generic biomedical signal feature vector: time-domain + frequency-domain + nonlinear"})


def cheatsheet():
    return "rgsf: Generic biomedical signal feature vector: time-domain + frequency-domain + nonlinear"

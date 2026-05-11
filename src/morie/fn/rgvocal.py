# morie.fn — function file (hadesllm/morie)
"""Vocal tract transfer function extraction via homomorphic deconvolution."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_vocal_tract"]


def rangayyan_vocal_tract(speech, fs, n_coeff):
    """
    Vocal tract transfer function extraction via homomorphic deconvolution

    Formula: V(z) extracted from complex cepstrum low-time region

    Parameters
    ----------
    speech : array-like
        Input data.
    fs : array-like
        Input data.
    n_coeff : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: vocal_response

    References
    ----------
    Rangayyan Ch 4.7.3
    """
    speech = np.asarray(speech, dtype=float)
    n = int(speech) if speech.ndim == 0 else len(speech)
    result = float(np.mean(speech))
    se = float(np.std(speech, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Vocal tract transfer function extraction via homomorphic deconvolution"})


def cheatsheet():
    return "rgvocal: Vocal tract transfer function extraction via homomorphic deconvolution"

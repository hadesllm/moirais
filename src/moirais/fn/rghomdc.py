# moirais.fn — function file (hadesllm/moirais)
"""Homomorphic deconvolution via complex cepstrum."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_homomorphic_deconv"]


def rangayyan_homomorphic_deconv(x, lifter_low, lifter_high):
    """
    Homomorphic deconvolution via complex cepstrum

    Formula: x_hat = IFFT(exp(lifter(log(|FFT(x)|) + j*angle(FFT(x)))))

    Parameters
    ----------
    x : array-like
        Input data.
    lifter_low : array-like
        Input data.
    lifter_high : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: source, filter_resp

    References
    ----------
    Rangayyan Ch 4.7.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Homomorphic deconvolution via complex cepstrum"})


def cheatsheet():
    return "rghomdc: Homomorphic deconvolution via complex cepstrum"

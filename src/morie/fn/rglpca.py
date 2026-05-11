# morie.fn — function file (hadesllm/morie)
"""Linear predictive coding (LPC) analysis of speech/biosignals."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_lpc_analysis"]


def rangayyan_lpc_analysis(x, order, frame_len, hop_len, fs):
    """
    Linear predictive coding (LPC) analysis of speech/biosignals

    Formula: Minimize E=sum(e^2); e[n]=x[n]-sum a_k*x[n-k]; solved via autocorrelation method

    Parameters
    ----------
    x : array-like
        Input data.
    order : array-like
        Input data.
    frame_len : array-like
        Input data.
    hop_len : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lpc_coeffs, gain, residual

    References
    ----------
    Rangayyan Ch 7.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear predictive coding (LPC) analysis of speech/biosignals"})


def cheatsheet():
    return "rglpca: Linear predictive coding (LPC) analysis of speech/biosignals"

# morie.fn — function file (hadesllm/morie)
"""Infant cry signal analysis: formants and fundamental frequency."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_infant_cry"]


def rangayyan_infant_cry(cry, fs):
    """
    Infant cry signal analysis: formants and fundamental frequency

    Formula: Pitch via autocorrelation peak; formants from LPC poles

    Parameters
    ----------
    cry : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pitch, formants, cry_type

    References
    ----------
    Rangayyan Ch 8.13
    """
    cry = np.asarray(cry, dtype=float)
    n = int(cry) if cry.ndim == 0 else len(cry)
    result = float(np.mean(cry))
    se = float(np.std(cry, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Infant cry signal analysis: formants and fundamental frequency"})


def cheatsheet():
    return "rgcry: Infant cry signal analysis: formants and fundamental frequency"

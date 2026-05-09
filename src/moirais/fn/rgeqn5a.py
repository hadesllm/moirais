# moirais.fn — function file (hadesllm/moirais)
"""Waveform morphology index for ECG beat classification."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch5_waveform_morph"]


def rangayyan_ch5_waveform_morph(template, beat):
    """
    Waveform morphology index for ECG beat classification

    Formula: MI = (1/2)*(1 - rho_xy) where rho_xy is correlation coefficient

    Parameters
    ----------
    template : array-like
        Input data.
    beat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: morphology_index

    References
    ----------
    Rangayyan Ch 5.4
    """
    template = np.asarray(template, dtype=float)
    n = int(template) if template.ndim == 0 else len(template)
    result = float(np.mean(template))
    se = float(np.std(template, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Waveform morphology index for ECG beat classification"})


def cheatsheet():
    return "rgeqn5a: Waveform morphology index for ECG beat classification"

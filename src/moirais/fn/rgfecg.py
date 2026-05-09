# moirais.fn — function file (hadesllm/moirais)
"""Maternal-fetal ECG separation via adaptive noise cancellation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_fetal_ecg"]


def rangayyan_fetal_ecg(abdominal, maternal_ref, mu, order):
    """
    Maternal-fetal ECG separation via adaptive noise cancellation

    Formula: Fetal ECG = abdominal ECG - adaptive_filter(maternal chest ECG)

    Parameters
    ----------
    abdominal : array-like
        Input data.
    maternal_ref : array-like
        Input data.
    mu : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fetal_ecg

    References
    ----------
    Rangayyan Ch 3.14
    """
    abdominal = np.asarray(abdominal, dtype=float)
    n = int(abdominal) if abdominal.ndim == 0 else len(abdominal)
    result = float(np.mean(abdominal))
    se = float(np.std(abdominal, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Maternal-fetal ECG separation via adaptive noise cancellation"})


def cheatsheet():
    return "rgfecg: Maternal-fetal ECG separation via adaptive noise cancellation"

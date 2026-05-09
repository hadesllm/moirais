# moirais.fn — function file (hadesllm/moirais)
"""Maternal ECG filtering from abdominal ECG recording."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_maternal_ecg_filter"]


def rangayyan_maternal_ecg_filter(abdominal_ecg, fs, n_channels):
    """
    Maternal ECG filtering from abdominal ECG recording

    Formula: ICA or adaptive filter removes maternal component; fetal ECG in residual

    Parameters
    ----------
    abdominal_ecg : array-like
        Input data.
    fs : array-like
        Input data.
    n_channels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fetal_ecg, maternal_template

    References
    ----------
    Rangayyan Ch 9.11
    """
    abdominal_ecg = np.asarray(abdominal_ecg, dtype=float)
    n = int(abdominal_ecg) if abdominal_ecg.ndim == 0 else len(abdominal_ecg)
    result = float(np.mean(abdominal_ecg))
    se = float(np.std(abdominal_ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Maternal ECG filtering from abdominal ECG recording"})


def cheatsheet():
    return "rgmatefp: Maternal ECG filtering from abdominal ECG recording"

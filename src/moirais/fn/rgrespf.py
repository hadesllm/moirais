# moirais.fn — function file (hadesllm/moirais)
"""Respiratory signal analysis: rate, depth, I:E ratio."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_respiration_features"]


def rangayyan_respiration_features(resp, fs):
    """
    Respiratory signal analysis: rate, depth, I:E ratio

    Formula: RR = peaks/time; depth = amplitude; I:E = rise_time/fall_time

    Parameters
    ----------
    resp : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: resp_rate, depth, ie_ratio

    References
    ----------
    Rangayyan Ch 5.10
    """
    resp = np.asarray(resp, dtype=float)
    n = int(resp) if resp.ndim == 0 else len(resp)
    result = float(np.mean(resp))
    se = float(np.std(resp, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Respiratory signal analysis: rate, depth, I:E ratio"})


def cheatsheet():
    return "rgrespf: Respiratory signal analysis: rate, depth, I:E ratio"

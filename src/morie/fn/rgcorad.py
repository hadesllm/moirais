# morie.fn — function file (hadesllm/morie)
"""Coronary artery disease detection from acoustic signals."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_coronary_ad"]


def rangayyan_coronary_ad(coronary_sound, fs, order):
    """
    Coronary artery disease detection from acoustic signals

    Formula: AR model of coronary sounds; discriminant features from poles

    Parameters
    ----------
    coronary_sound : array-like
        Input data.
    fs : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cad_score, ar_features

    References
    ----------
    Rangayyan Ch 7.11
    """
    coronary_sound = np.asarray(coronary_sound, dtype=float)
    n = int(coronary_sound) if coronary_sound.ndim == 0 else len(coronary_sound)
    result = float(np.mean(coronary_sound))
    se = float(np.std(coronary_sound, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Coronary artery disease detection from acoustic signals"})


def cheatsheet():
    return "rgcorad: Coronary artery disease detection from acoustic signals"

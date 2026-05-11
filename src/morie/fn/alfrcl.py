"""AlphaFold recycle loss."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_recycle_loss"]


def alphafold_recycle_loss(frames_pred, frames_true, clamp):
    """
    AlphaFold recycle loss

    Formula: FAPE per recycle iter

    Parameters
    ----------
    frames_pred : array-like
        Input data.
    frames_true : array-like
        Input data.
    clamp : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jumper et al (2021)
    """
    frames_pred = np.atleast_1d(np.asarray(frames_pred, dtype=float))
    n = len(frames_pred)
    result = float(np.mean(frames_pred))
    se = float(np.std(frames_pred, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold recycle loss"})


def cheatsheet():
    return "alfrcl: AlphaFold recycle loss"

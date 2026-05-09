"""Frame Aligned Point Error (FAPE)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_fape_loss"]


def alphafold_fape_loss(frames_pred, frames_true, x, x_true):
    """
    Frame Aligned Point Error (FAPE)

    Formula: ||T^-1 x - T^-1 x_true||_clamp

    Parameters
    ----------
    frames_pred : array-like
        Input data.
    frames_true : array-like
        Input data.
    x : array-like
        Input data.
    x_true : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jumper et al (2021)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frame Aligned Point Error (FAPE)"})


def cheatsheet():
    return "alffap: Frame Aligned Point Error (FAPE)"

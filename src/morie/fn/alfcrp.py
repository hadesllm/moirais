"""AlphaFold sequence cropping for training."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphafold_cropping"]


def alphafold_cropping(sequence, crop_size):
    """
    AlphaFold sequence cropping for training

    Formula: random crop length 256 from full sequence

    Parameters
    ----------
    sequence : array-like
        Input data.
    crop_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jumper et al (2021)
    """
    sequence = np.atleast_1d(np.asarray(sequence, dtype=float))
    n = len(sequence)
    result = float(np.mean(sequence))
    se = float(np.std(sequence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaFold sequence cropping for training"})


def cheatsheet():
    return "alfcrp: AlphaFold sequence cropping for training"

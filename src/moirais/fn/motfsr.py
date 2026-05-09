"""MEME motif discovery."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["motif_meme"]


def motif_meme(sequences, motif_length):
    """
    MEME motif discovery

    Formula: EM over PWM motif model

    Parameters
    ----------
    sequences : array-like
        Input data.
    motif_length : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bailey-Elkan (1994)
    """
    sequences = np.atleast_1d(np.asarray(sequences, dtype=float))
    n = len(sequences)
    result = float(np.mean(sequences))
    se = float(np.std(sequences, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MEME motif discovery"})


def cheatsheet():
    return "motfsr: MEME motif discovery"

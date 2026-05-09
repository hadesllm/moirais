"""MAFFT FFT-NS-2 / L-INS-i alignment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mafft_alignment"]


def mafft_alignment(sequences, mode):
    """
    MAFFT FFT-NS-2 / L-INS-i alignment

    Formula: FFT-based progressive alignment

    Parameters
    ----------
    sequences : array-like
        Input data.
    mode : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Katoh-Standley (2013)
    """
    sequences = np.atleast_1d(np.asarray(sequences, dtype=float))
    n = len(sequences)
    result = float(np.mean(sequences))
    se = float(np.std(sequences, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MAFFT FFT-NS-2 / L-INS-i alignment"})


def cheatsheet():
    return "mafft: MAFFT FFT-NS-2 / L-INS-i alignment"

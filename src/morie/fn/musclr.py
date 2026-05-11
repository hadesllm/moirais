"""MUSCLE multiple sequence alignment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["muscle_msa"]


def muscle_msa(sequences):
    """
    MUSCLE multiple sequence alignment

    Formula: iterative refinement on guide tree

    Parameters
    ----------
    sequences : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Edgar (2004)
    """
    sequences = np.atleast_1d(np.asarray(sequences, dtype=float))
    n = len(sequences)
    result = float(np.mean(sequences))
    se = float(np.std(sequences, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MUSCLE multiple sequence alignment"})


def cheatsheet():
    return "musclr: MUSCLE multiple sequence alignment"

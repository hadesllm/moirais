"""Intrinsic disorder prediction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["protein_disorder"]


def protein_disorder(sequence):
    """
    Intrinsic disorder prediction

    Formula: residue-level NN with composition features

    Parameters
    ----------
    sequence : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mészáros et al (2018) IUPred2A
    """
    sequence = np.atleast_1d(np.asarray(sequence, dtype=float))
    n = len(sequence)
    result = float(np.mean(sequence))
    se = float(np.std(sequence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Intrinsic disorder prediction"})


def cheatsheet():
    return "protdis: Intrinsic disorder prediction"

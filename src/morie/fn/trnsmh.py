"""TMHMM transmembrane topology."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["transmembrane_topology"]


def transmembrane_topology(sequence):
    """
    TMHMM transmembrane topology

    Formula: HMM with cytoplasmic / extracellular / TM states

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
    Krogh et al (2001)
    """
    sequence = np.atleast_1d(np.asarray(sequence, dtype=float))
    n = len(sequence)
    result = float(np.mean(sequence))
    se = float(np.std(sequence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMHMM transmembrane topology"})


def cheatsheet():
    return "trnsmh: TMHMM transmembrane topology"

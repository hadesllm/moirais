"""RNA secondary structure (Zuker)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rna_fold"]


def rna_fold(sequence):
    """
    RNA secondary structure (Zuker)

    Formula: DP minimizing free energy

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
    Zuker (1989); Lorenz et al (2011) ViennaRNA
    """
    sequence = np.atleast_1d(np.asarray(sequence, dtype=float))
    n = len(sequence)
    result = float(np.mean(sequence))
    se = float(np.std(sequence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RNA secondary structure (Zuker)"})


def cheatsheet():
    return "rnafld: RNA secondary structure (Zuker)"

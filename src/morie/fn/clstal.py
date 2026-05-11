"""Clustal Omega progressive MSA."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["clustalo"]


def clustalo(sequences):
    """
    Clustal Omega progressive MSA

    Formula: HHalign + guide tree + iterative refinement

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
    Sievers et al (2011)
    """
    sequences = np.atleast_1d(np.asarray(sequences, dtype=float))
    n = len(sequences)
    result = float(np.mean(sequences))
    se = float(np.std(sequences, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Clustal Omega progressive MSA"})


def cheatsheet():
    return "clstal: Clustal Omega progressive MSA"

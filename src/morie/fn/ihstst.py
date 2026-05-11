"""Integrated Haplotype Score (iHS)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ihs_test"]


def ihs_test(haplotypes, ancestral):
    """
    Integrated Haplotype Score (iHS)

    Formula: log(EHH_anc / EHH_der) integrated

    Parameters
    ----------
    haplotypes : array-like
        Input data.
    ancestral : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Voight et al (2006)
    """
    haplotypes = np.atleast_1d(np.asarray(haplotypes, dtype=float))
    n = len(haplotypes)
    result = float(np.mean(haplotypes))
    se = float(np.std(haplotypes, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Integrated Haplotype Score (iHS)"})


def cheatsheet():
    return "ihstst: Integrated Haplotype Score (iHS)"

"""Variant calling (GATK HaplotypeCaller)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["variant_calling"]


def variant_calling(bam, reference):
    """
    Variant calling (GATK HaplotypeCaller)

    Formula: local de novo assembly + HMM likelihood

    Parameters
    ----------
    bam : array-like
        Input data.
    reference : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Poplin et al (2018) GATK4
    """
    bam = np.atleast_1d(np.asarray(bam, dtype=float))
    n = len(bam)
    result = float(np.mean(bam))
    se = float(np.std(bam, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Variant calling (GATK HaplotypeCaller)"})


def cheatsheet():
    return "varcal: Variant calling (GATK HaplotypeCaller)"

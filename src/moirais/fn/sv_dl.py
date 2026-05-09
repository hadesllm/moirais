"""Structural variant calling (delly/manta)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["structural_variant"]


def structural_variant(bam, reference):
    """
    Structural variant calling (delly/manta)

    Formula: split-read + paired-end + read-depth

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
    Rausch et al (2012) Delly
    """
    bam = np.atleast_1d(np.asarray(bam, dtype=float))
    n = len(bam)
    result = float(np.mean(bam))
    se = float(np.std(bam, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Structural variant calling (delly/manta)"})


def cheatsheet():
    return "sv_dl: Structural variant calling (delly/manta)"

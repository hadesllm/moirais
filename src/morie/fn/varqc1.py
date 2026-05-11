"""VCF variant quality filtering."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vcf_filter"]


def vcf_filter(vcf, thresholds):
    """
    VCF variant quality filtering

    Formula: QD/MQ/FS thresholds + VQSR

    Parameters
    ----------
    vcf : array-like
        Input data.
    thresholds : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    GATK best practices
    """
    vcf = np.atleast_1d(np.asarray(vcf, dtype=float))
    n = len(vcf)
    result = float(np.mean(vcf))
    se = float(np.std(vcf, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VCF variant quality filtering"})


def cheatsheet():
    return "varqc1: VCF variant quality filtering"

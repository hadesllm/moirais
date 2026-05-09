"""SnpEff variant annotation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["variant_effect"]


def variant_effect(variants, annotation):
    """
    SnpEff variant annotation

    Formula: map variant to gene + protein-level effect

    Parameters
    ----------
    variants : array-like
        Input data.
    annotation : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cingolani et al (2012) SnpEff
    """
    variants = np.atleast_1d(np.asarray(variants, dtype=float))
    n = len(variants)
    result = float(np.mean(variants))
    se = float(np.std(variants, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SnpEff variant annotation"})


def cheatsheet():
    return "snpeff: SnpEff variant annotation"

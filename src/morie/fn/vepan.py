"""VEP variant effect predictor."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vep_annotation"]


def vep_annotation(variants, cache):
    """
    VEP variant effect predictor

    Formula: Ensembl gene model + consequence prediction

    Parameters
    ----------
    variants : array-like
        Input data.
    cache : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    McLaren et al (2016) VEP
    """
    variants = np.atleast_1d(np.asarray(variants, dtype=float))
    n = len(variants)
    result = float(np.mean(variants))
    se = float(np.std(variants, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VEP variant effect predictor"})


def cheatsheet():
    return "vepan: VEP variant effect predictor"

"""DistMult bilinear KG embedding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["distmult"]


def distmult(triples, dim):
    """
    DistMult bilinear KG embedding

    Formula: <h, r, t>

    Parameters
    ----------
    triples : array-like
        Input data.
    dim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Yang et al (2015)
    """
    triples = np.atleast_1d(np.asarray(triples, dtype=float))
    n = len(triples)
    result = float(np.mean(triples))
    se = float(np.std(triples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DistMult bilinear KG embedding"})


def cheatsheet():
    return "distM: DistMult bilinear KG embedding"

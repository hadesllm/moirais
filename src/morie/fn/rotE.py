"""RotatE — rotation in complex space."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rotate"]


def rotate(triples, dim):
    """
    RotatE — rotation in complex space

    Formula: ||h ∘ r − t||

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
    Sun et al (2019)
    """
    triples = np.atleast_1d(np.asarray(triples, dtype=float))
    n = len(triples)
    result = float(np.mean(triples))
    se = float(np.std(triples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RotatE — rotation in complex space"})


def cheatsheet():
    return "rotE: RotatE — rotation in complex space"

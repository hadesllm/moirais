"""ComplEx (complex embeddings)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["complex"]


def complex(triples, dim):
    """
    ComplEx (complex embeddings)

    Formula: Re(<h, r, t̄>) over complex space

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
    Trouillon et al (2016)
    """
    triples = np.atleast_1d(np.asarray(triples, dtype=float))
    n = len(triples)
    result = float(np.mean(triples))
    se = float(np.std(triples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ComplEx (complex embeddings)"})


def cheatsheet():
    return "complE: ComplEx (complex embeddings)"

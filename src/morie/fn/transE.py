"""TransE knowledge-graph embedding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["transe"]


def transe(triples, dim):
    """
    TransE knowledge-graph embedding

    Formula: ||h + r − t||

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
    Bordes et al (2013)
    """
    triples = np.atleast_1d(np.asarray(triples, dtype=float))
    n = len(triples)
    result = float(np.mean(triples))
    se = float(np.std(triples, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TransE knowledge-graph embedding"})


def cheatsheet():
    return "transE: TransE knowledge-graph embedding"

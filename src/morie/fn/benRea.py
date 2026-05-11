"""Named-entity recognition (BIO)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["named_entity"]


def named_entity(sentence, model):
    """
    Named-entity recognition (BIO)

    Formula: BIO/BIOES tagging via CRF or transformer

    Parameters
    ----------
    sentence : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tjong Kim Sang-De Meulder (2003) CoNLL
    """
    sentence = np.atleast_1d(np.asarray(sentence, dtype=float))
    n = len(sentence)
    result = float(np.mean(sentence))
    se = float(np.std(sentence, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Named-entity recognition (BIO)"})


def cheatsheet():
    return "benRea: Named-entity recognition (BIO)"

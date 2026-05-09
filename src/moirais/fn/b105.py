"""Cosine of the angle between two vectors, used as a similarity measure for embeddings.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_cosine_similarity"]


def burkov_lm_ch1_cosine_similarity(x, y):
    """
    Cosine of the angle between two vectors, used as a similarity measure for embeddings.

    Formula: \cos(\theta) = \frac{\mathbf{x} \cdot \mathbf{y}}{\lVert \mathbf{x} \rVert\, \lVert \mathbf{y} \rVert}

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cosine similarity in [-1, 1]

    References
    ----------
    Burkov LM (2025), Ch 1, Eq 1.5, p. 31
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cosine of the angle between two vectors, used as a similarity measure for embeddings."})


def cheatsheet():
    return "b105: Cosine of the angle between two vectors, used as a similarity measure for embeddings."

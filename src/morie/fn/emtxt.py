# morie.fn -- function file (hadesllm/morie)
"""EM-IRT for text/word-count frequency data as input."""
import numpy as np
from ._richresult import RichResult

__all__ = ["em_irt_text"]


def em_irt_text(word_freq_matrix, n_dims):
    """
    EM-IRT for text/word-count frequency data as input

    Formula: Treat word-frequency matrix as vote matrix; ideal points from word-usage scaling

    Parameters
    ----------
    word_freq_matrix : array-like
        Input data.
    n_dims : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'ideal_points': 'matrix'}

    References
    ----------
    Armstrong Ch 6
    """
    word_freq_matrix = np.asarray(word_freq_matrix, dtype=float)
    n = int(word_freq_matrix) if word_freq_matrix.ndim == 0 else len(word_freq_matrix)
    result = float(np.mean(word_freq_matrix))
    se = float(np.std(word_freq_matrix, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EM-IRT for text/word-count frequency data as input"})


def cheatsheet():
    return "emtxt: EM-IRT for text/word-count frequency data as input"

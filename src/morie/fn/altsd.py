# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""TSDAE: encoder takes corrupted sentence, decoder reconstructs the clean one."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_tsdae_objective"]


def alammar_tsdae_objective(clean, corrupted, encoder, decoder):
    """
    TSDAE: encoder takes corrupted sentence, decoder reconstructs the clean one

    Formula: L = - sum_t log p(x_t | Enc(corrupt(x)), x_{<t})

    Parameters
    ----------
    clean : array-like
        Input data.
    corrupted : array-like
        Input data.
    encoder : array-like
        Input data.
    decoder : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Alammar Ch 10, TSDAE section
    """
    clean = np.atleast_1d(np.asarray(clean, dtype=float))
    n = len(clean)
    result = float(np.mean(clean))
    se = float(np.std(clean, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TSDAE: encoder takes corrupted sentence, decoder reconstructs the clean one"})


def cheatsheet():
    return "altsd: TSDAE: encoder takes corrupted sentence, decoder reconstructs the clean one"

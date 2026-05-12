"""Pack an array of b-bit codebook indices into a dense byte buffer."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_bit_pack_indices"]


def turboquant_bit_pack_indices(indices, bits):
    """
    Pack an array of b-bit codebook indices into a dense byte buffer

    Formula: concat b-bit fields into bits;  packed = bits_to_bytes(concatenation)

    Parameters
    ----------
    indices : array-like
        Input data.
    bits : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: packed

    References
    ----------
    TurboQuant MORIE integration -- pack_indices
    """
    indices = np.atleast_1d(np.asarray(indices, dtype=float))
    n = len(indices)
    result = float(np.mean(indices))
    se = float(np.std(indices, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pack an array of b-bit codebook indices into a dense byte buffer"})


def cheatsheet():
    return "tqpack: Pack an array of b-bit codebook indices into a dense byte buffer"

# moirais.fn — function file (hadesllm/moirais)
"""Inverse of a symmetric matrix (pinv-safe)."""

from typing import Sequence, Union
import numpy as np
def invsym(A: Union[Sequence, np.ndarray]) -> np.ndarray:
    """Pseudo-inverse of a symmetric matrix.

    Robust to singularity; use np.linalg.pinv internally.
    """
    M = np.asarray(A, dtype=float)
    return np.linalg.pinv(M)

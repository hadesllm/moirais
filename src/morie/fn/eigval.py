# morie.fn -- function file (hadesllm/morie)
"""Eigenvalues of a symmetric matrix."""

from typing import Sequence, Union
import numpy as np
def eigval(A: Union[Sequence, np.ndarray]) -> np.ndarray:
    """Eigenvalues of a symmetric matrix (ascending order)."""
    M = np.asarray(A, dtype=float)
    return np.linalg.eigvalsh(M)

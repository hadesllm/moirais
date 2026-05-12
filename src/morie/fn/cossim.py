# morie.fn -- function file (hadesllm/morie)
"""Cosine similarity."""

from typing import Sequence, Union
import numpy as np
def cossim(x: Union[Sequence, np.ndarray],
           y: Union[Sequence, np.ndarray]) -> float:
    """Cosine similarity: x·y / (||x|| ||y||)."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("zero-norm vector.")
    return float(np.dot(a, b) / (na * nb))

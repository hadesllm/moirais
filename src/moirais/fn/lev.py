# moirais.fn — function file (hadesllm/moirais)
"""Leverage diagonal."""

from typing import Sequence, Union
import numpy as np
def lev(X: Union[Sequence, np.ndarray]) -> np.ndarray:
    """Hat-matrix diagonals (leverage values).

    h = X (X'X)⁻¹ X'; returns just the diagonal.
    """
    X = np.asarray(X, dtype=float)
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    return np.diag(H)

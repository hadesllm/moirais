# moirais.fn — function file (hadesllm/moirais)
"""Matrix trace (sum of diagonal)."""

from typing import Sequence, Union
import numpy as np
def trace(A: Union[Sequence, np.ndarray]) -> float:
    """Trace of a square matrix: Σᵢ Aᵢᵢ."""
    M = np.asarray(A, dtype=float)
    if M.shape[0] != M.shape[1]:
        raise ValueError("matrix must be square.")
    return float(np.trace(M))

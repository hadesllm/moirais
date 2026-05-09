# moirais.fn — function file (hadesllm/moirais)
"""Euclidean distance."""

from typing import Sequence, Union
import numpy as np
def eucldst(x: Union[Sequence, np.ndarray],
            y: Union[Sequence, np.ndarray]) -> float:
    """Euclidean (L₂) distance between vectors x and y."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    return float(np.sqrt(np.sum((a - b) ** 2)))

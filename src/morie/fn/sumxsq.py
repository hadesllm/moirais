"""Σxᵢ² -- sum of squares (atomic primitive)."""
from typing import Sequence, Union
import numpy as np

def sumxsq(x: Union[Sequence[float], np.ndarray]) -> float:
    """Σᵢ xᵢ² -- uncorrected sum of squares.

    Atomic primitive used by variance, regression SS, F-stats, etc.
    """
    a = np.asarray(x, dtype=float)
    return float(np.sum(a * a))

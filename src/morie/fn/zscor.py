# morie.fn -- function file (hadesllm/morie)
"""Z-score / standard score."""

from typing import Sequence, Union
import numpy as np
def zscor(x: Union[Sequence, np.ndarray]) -> np.ndarray:
    """Z-score: (x − x̄) / s. Sample-SD (ddof=1)."""
    a = np.asarray(x, dtype=float)
    sd = float(a.std(ddof=1))
    if sd == 0:
        raise ValueError("zero variance.")
    return (a - a.mean()) / sd

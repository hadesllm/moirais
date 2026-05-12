# morie.fn -- function file (hadesllm/morie)
"""Range normalization to [0, 1]."""

from typing import Sequence, Union
import numpy as np
def rngnrm(x: Union[Sequence, np.ndarray]) -> np.ndarray:
    """Min-max scale x to [0, 1]: (x − min) / (max − min)."""
    a = np.asarray(x, dtype=float)
    mn, mx = a.min(), a.max()
    if mx == mn:
        raise ValueError("all values are identical.")
    return (a - mn) / (mx - mn)

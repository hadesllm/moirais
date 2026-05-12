# morie.fn -- function file (hadesllm/morie)
"""Total sum of squares."""

from typing import Sequence, Union
import numpy as np

def sst(y: Union[Sequence[float], np.ndarray]) -> float:
    """Total sum of squares: Σᵢ (yᵢ − ȳ)².

    Decomposition: SST = SSR + SSE. Used by R², ANOVA F-stat, etc.
    """
    a = np.asarray(y, dtype=float)
    return float(np.sum((a - a.mean()) ** 2))

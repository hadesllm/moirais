# moirais.fn — function file (hadesllm/moirais)
"""Leave-one-out jackknife estimates."""

from typing import Sequence, Union, Callable
import numpy as np
def jackone(x: Union[Sequence, np.ndarray], stat: Callable) -> np.ndarray:
    """Leave-one-out jackknife: stat applied to each n−1-subset.

    Returns array of n estimates. Use to compute jackknife SE.
    """
    a = np.asarray(x)
    n = a.size
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = stat(np.delete(a, i))
    return out

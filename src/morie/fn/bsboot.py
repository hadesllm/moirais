# morie.fn — function file (hadesllm/morie)
"""Bootstrap resamples of a statistic."""

from typing import Sequence, Union, Callable
import numpy as np
def bsboot(x: Union[Sequence, np.ndarray],
           stat: Callable, n_boot: int = 1000, seed: int = 42) -> np.ndarray:
    """Generate `n_boot` bootstrap statistics by resampling x with
    replacement.

    Returns an array of n_boot estimates. Compute CI / SE downstream.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(x)
    n = a.size
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = stat(a[idx])
    return out

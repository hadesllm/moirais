# moirais.fn — function file (hadesllm/moirais)
"""Autocorrelation at lag k."""

from typing import Sequence, Union
import numpy as np
def autocor(x: Union[Sequence, np.ndarray], k: int = 1) -> float:
    """Sample autocorrelation at lag k.

    ρ̂(k) = γ̂(k) / γ̂(0)
    """
    from .autocov import autocov as _ac
    g0 = _ac(x, 0)
    if g0 == 0:
        raise ValueError("zero variance — autocorrelation undefined.")
    return _ac(x, k) / g0

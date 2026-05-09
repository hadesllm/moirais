# moirais.fn — function file (hadesllm/moirais)
"""Autocovariance at lag k."""

from typing import Sequence, Union
import numpy as np
def autocov(x: Union[Sequence, np.ndarray], k: int = 1) -> float:
    """Sample autocovariance at lag k.

    γ̂(k) = (1/n) Σₜ (xₜ − x̄)(xₜ₊ₖ − x̄)
    """
    a = np.asarray(x, dtype=float)
    n = a.size
    if k < 0 or k >= n:
        raise ValueError("require 0 ≤ k < n.")
    mu = a.mean()
    return float(((a[:n - k] - mu) * (a[k:] - mu)).sum() / n)

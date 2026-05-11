# morie.fn — function file (hadesllm/morie)
"""Softmax over a vector."""

from typing import Sequence, Union
import numpy as np
def softmx(x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Softmax: σᵢ(x) = exp(xᵢ) / Σⱼ exp(xⱼ).

    Numerically stable — subtracts max(x) before exponentiating.
    """
    a = np.asarray(x, dtype=float)
    e = np.exp(a - a.max())
    return e / e.sum()

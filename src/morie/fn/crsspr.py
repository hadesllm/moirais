# morie.fn -- function file (hadesllm/morie)
"""Cross product of centered vectors."""

from typing import Sequence, Union
import numpy as np
def crsspr(x: Union[Sequence, np.ndarray],
           y: Union[Sequence, np.ndarray]) -> float:
    """Σᵢ (xᵢ − x̄)(yᵢ − ȳ) -- cross-product of centered series.

    Numerator of covariance and Pearson r.
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    return float(np.sum((a - a.mean()) * (b - b.mean())))

# moirais.fn — function file (hadesllm/moirais)
"""Manhattan (L₁) distance."""

from typing import Sequence, Union
import numpy as np
def manhdst(x: Union[Sequence, np.ndarray],
            y: Union[Sequence, np.ndarray]) -> float:
    """Manhattan (L₁) distance: Σ |xᵢ − yᵢ|."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    return float(np.sum(np.abs(a - b)))

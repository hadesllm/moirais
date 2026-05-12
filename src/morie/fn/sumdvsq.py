"""Σ(xᵢ - x̄)² -- sum of squared deviations from the mean (atomic)."""
from typing import Sequence, Union
import numpy as np

def sumdvsq(x: Union[Sequence[float], np.ndarray]) -> float:
    """Sum of squared deviations: Σᵢ (xᵢ − x̄)².

    The numerator of sample variance. Used by SST, ANOVA, regression
    centering. Power-of-two cancellation is cheaper here than in scaled
    forms -- keep this primitive available without the (n−1) divisor.
    """
    a = np.asarray(x, dtype=float)
    return float(np.sum((a - a.mean()) ** 2))

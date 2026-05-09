# moirais.fn — function file (hadesllm/moirais)
"""Regression sum of squares."""

from typing import Sequence, Union
import numpy as np

def ssr(y_pred: Union[Sequence[float], np.ndarray],
        y_true: Union[Sequence[float], np.ndarray]) -> float:
    """Regression sum of squares: Σᵢ (ŷᵢ − ȳ)².

    Variation in y captured by the fitted model.
    """
    yp = np.asarray(y_pred, dtype=float)
    yt = np.asarray(y_true, dtype=float)
    return float(np.sum((yp - yt.mean()) ** 2))

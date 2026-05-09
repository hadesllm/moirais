# moirais.fn — function file (hadesllm/moirais)
"""Error sum of squares."""

from typing import Sequence, Union
import numpy as np

def sse(y_true: Union[Sequence[float], np.ndarray],
        y_pred: Union[Sequence[float], np.ndarray]) -> float:
    """Error (residual) sum of squares: Σᵢ (yᵢ − ŷᵢ)².

    Penalty term in OLS minimisation.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sum((yt - yp) ** 2))

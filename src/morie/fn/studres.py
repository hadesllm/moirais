# morie.fn — function file (hadesllm/morie)
"""Studentized residual."""

from typing import Sequence, Union
import numpy as np
def studres(y: Union[Sequence, np.ndarray],
            yhat: Union[Sequence, np.ndarray],
            leverage: Union[Sequence, np.ndarray]) -> np.ndarray:
    """Internally-studentized residuals.

    rᵢ_studentized = (yᵢ − ŷᵢ) / (s × sqrt(1 − hᵢᵢ))
    where s = sqrt(SSE / (n − p)).
    """
    y = np.asarray(y, dtype=float)
    yh = np.asarray(yhat, dtype=float)
    h = np.asarray(leverage, dtype=float)
    resid = y - yh
    n = y.size
    p_eff = sum(h)  # trace of hat matrix
    s = float(np.sqrt(np.sum(resid ** 2) / max(n - p_eff, 1)))
    return resid / (s * np.sqrt(np.maximum(1 - h, 1e-12)))

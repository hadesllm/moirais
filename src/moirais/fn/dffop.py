# moirais.fn — function file (hadesllm/moirais)
"""k-th difference operator."""

from typing import Sequence, Union
import numpy as np
def dffop(x: Union[Sequence, np.ndarray], k: int = 1) -> np.ndarray:
    """k-th-order difference. ∇x = xₜ − xₜ₋₁; ∇²x = ∇(∇x); etc.

    Used to remove trends before stationarity tests / ARIMA fitting.
    """
    a = np.asarray(x, dtype=float)
    return np.diff(a, n=k)

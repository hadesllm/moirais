# morie.fn -- function file (hadesllm/morie)
"""Bootstrap percentile CI."""

from typing import Sequence, Union
import numpy as np
def bsperc(boot_estimates: Union[Sequence[float], np.ndarray],
           conf: float = 0.95) -> tuple:
    """Percentile bootstrap confidence interval (lo, hi).

    Crude but widely-used. Pass output of `bsboot()`.
    """
    a = np.asarray(boot_estimates, dtype=float)
    if not 0 < conf < 1:
        raise ValueError("conf in (0, 1).")
    lo_q = (1 - conf) / 2
    hi_q = 1 - lo_q
    return (float(np.quantile(a, lo_q)), float(np.quantile(a, hi_q)))

# morie.fn -- function file (hadesllm/morie)
"""Shannon entropy with R-style verbose result."""

import math
from typing import Sequence, Union
import numpy as np


def shanon(p: Union[Sequence[float], np.ndarray], base: float = 2.0):
    """Shannon entropy H(p) = -Σ p_i log_b(p_i)."""
    from ._richresult import RichResult
    a = np.asarray(p, dtype=float)
    if np.any(a < 0):
        raise ValueError("p must be non-negative.")
    s = a.sum()
    if s == 0:
        raise ValueError("p sums to zero.")
    a = a / s
    nz = a[a > 0]
    H = float(-np.sum(nz * np.log(nz)) / math.log(base))
    H_max = math.log(a.size) / math.log(base)  # uniform = max entropy
    norm = H / H_max if H_max > 0 else float("nan")
    return RichResult(
        title="Shannon entropy",
        summary_lines=[
            (f"H(p), base {base}", H),
            (f"Max H (uniform on {a.size})", H_max),
            ("Normalized H / H_max", norm),
            ("Support size", int(a.size)),
            ("Effective alphabet", float(2 ** H if base == 2 else math.exp(H))),
        ],
        interpretation=(f"H={H:.4f} {'bits' if base == 2 else 'units'}; "
                        f"normalized = {norm:.3f} (1.0 = uniform)."),
        payload={"value": H, "statistic": H, "max_entropy": H_max, "normalized": norm},
    )

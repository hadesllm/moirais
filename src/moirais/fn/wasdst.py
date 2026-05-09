# moirais.fn — function file (hadesllm/moirais)
"""1D Wasserstein distance with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import wasserstein_distance


def wasdst(u: Union[Sequence, np.ndarray],
           v: Union[Sequence, np.ndarray]):
    """1D Wasserstein-1 (earth mover's) distance."""
    from ._richresult import RichResult
    a = np.asarray(u, dtype=float); b = np.asarray(v, dtype=float)
    d = float(wasserstein_distance(a, b))
    return RichResult(
        title="1D Wasserstein-1 distance (earth mover's)",
        summary_lines=[
            ("W_1(P, Q)", d),
            ("n(u)", int(a.size)), ("n(v)", int(b.size)),
            ("Mean(u)", float(a.mean())), ("Mean(v)", float(b.mean())),
            ("Mean diff", float(a.mean() - b.mean())),
        ],
        interpretation=("Wasserstein-1 = ∫|F_P - F_Q| dx. Symmetric. Minimum "
                        "transport cost to transform P into Q. More robust to "
                        "disjoint supports than KL (`kldivg`)."),
        payload={"value": d, "statistic": d},
    )

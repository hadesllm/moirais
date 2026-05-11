# morie.fn — function file (hadesllm/morie)
"""Log-likelihood with R-style verbose result."""

from typing import Callable, Sequence, Union
import numpy as np


def loglik(x: Union[Sequence[float], np.ndarray],
           pdf: Callable[[float], float]):
    """Sum of log-pdfs evaluated at each observation."""
    from ._richresult import RichResult
    a = np.asarray(x, dtype=float)
    densities = np.array([pdf(v) for v in a], dtype=float)
    if np.any(densities <= 0):
        zero_idx = np.where(densities <= 0)[0]
        raise ValueError(f"pdf returned zero/negative density at index "
                         f"{zero_idx[0]} (value={a[zero_idx[0]]}).")
    log_d = np.log(densities)
    ll = float(np.sum(log_d))
    return RichResult(
        title="Log-likelihood",
        summary_lines=[
            ("Log-likelihood", ll),
            ("n observations", int(a.size)),
            ("Mean log-density", float(log_d.mean())),
            ("Min log-density", float(log_d.min())),
            ("Max log-density", float(log_d.max())),
        ],
        interpretation=("Higher (less negative) is better fit. Compare via "
                        "AIC (`akike`), BIC (`bayic`), or LR test (`lrtst`)."),
        payload={"value": ll, "statistic": ll},
    )

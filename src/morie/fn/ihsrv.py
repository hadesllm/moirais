# morie.fn — function file (hadesllm/morie)
"""Indigenous health service access disparity."""

import numpy as np
import scipy.stats as stats

from ._containers import ESRes


def indigenous_service_access(
    access_indigenous: float,
    access_general: float,
    n_indigenous: int = 0,
    n_general: int = 0,
    confidence: float = 0.95,
) -> ESRes:
    """Compute service access disparity ratio.

    Parameters
    ----------
    access_indigenous : float
        Proportion with access (0-1) in Indigenous population.
    access_general : float
        Proportion with access in general population.
    n_indigenous, n_general : int
        Sample sizes (for CI, optional).
    confidence : float

    Returns
    -------
    ESRes
    """
    ratio = access_indigenous / access_general if access_general > 0 else float("inf")
    gap = access_general - access_indigenous

    extra = {"gap": float(gap), "access_indigenous": float(access_indigenous), "access_general": float(access_general)}

    ci_lo, ci_hi = None, None
    if n_indigenous > 0 and n_general > 0:
        se_log = np.sqrt(
            (1 - access_indigenous) / (access_indigenous * n_indigenous + 1e-10)
            + (1 - access_general) / (access_general * n_general + 1e-10)
        )
        z = stats.norm.ppf((1 + confidence) / 2)
        lr = np.log(max(ratio, 1e-10))
        ci_lo = float(np.exp(lr - z * se_log))
        ci_hi = float(np.exp(lr + z * se_log))

    return ESRes(
        measure="service_access_ratio",
        estimate=float(ratio),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        extra=extra,
    )


ihsrv = indigenous_service_access


def cheatsheet() -> str:
    return "indigenous_service_access({}) -> Indigenous health service access disparity."

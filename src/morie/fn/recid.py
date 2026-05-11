# morie.fn — function file (hadesllm/morie)
"""General recidivism rate. 'There is always a bigger fish.' -- Qui-Gon Jinn"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import CrimeResult


def recidivism_rate(
    n_recidivists: int,
    n_released: int,
    confidence: float = 0.95,
) -> CrimeResult:
    """Recidivism rate as a proportion with Wilson CI.

    Parameters
    ----------
    n_recidivists : int
        Number who re-offended.
    n_released : int
        Total number released from custody.
    confidence : float, default 0.95

    Returns
    -------
    CrimeResult
    """
    if n_released <= 0:
        raise ValueError("n_released must be positive")
    p_hat = n_recidivists / n_released
    z = _st.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n_released
    centre = (p_hat + z**2 / (2 * n_released)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n_released + z**2 / (4 * n_released**2)) / denom
    return CrimeResult(
        name="Recidivism rate",
        rate=float(p_hat),
        ci_lower=float(max(0, centre - margin)),
        ci_upper=float(min(1, centre + margin)),
        n=n_recidivists,
        population=n_released,
        extra={"confidence": confidence},
    )


recid = recidivism_rate


def cheatsheet() -> str:
    return "recidivism_rate({}) -> General recidivism rate. 'There is always a bigger fish.' --"

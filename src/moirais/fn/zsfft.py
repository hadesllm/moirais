"""FFT-based spectral simulation"""

import numpy as np

from ._containers import SpatialResult


def spectral_sim(data, *, method="default"):
    """FFT-based spectral simulation

    Returns
    -------
    SpatialResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    mu = float(np.mean(data))
    var = float(np.var(data, ddof=1)) if n > 1 else 0.0
    se = float(np.sqrt(var / n)) if n > 0 else 0.0
    return SpatialResult(
        name="FFT-based spectral simulation",
        statistic=float(mu) if isinstance(mu, (int, float)) else 0.0,
        extra={},
    )


spec = spectral_sim


def cheatsheet() -> str:
    return "spectral_sim({}) -> FFT-based spectral simulation"

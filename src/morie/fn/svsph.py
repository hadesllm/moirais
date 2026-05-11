"""Spatial phase transition (chaos/order)"""

import numpy as np

from ._containers import DescriptiveResult


def spatial_phase(data, *, method="default"):
    """Spatial phase transition (chaos/order)

    Parameters
    ----------
    data : array-like
        Input data values.
    method : str
        Computation method.

    Returns
    -------
    DescriptiveResult
    """
    data = np.asarray(data, dtype=float)
    n = int(data) if data.ndim == 0 else len(data)
    if n < 2:
        val = 0.0
    else:
        sorted_d = np.sort(data)
        gaps = np.diff(sorted_d)
        mean_gap = float(np.mean(gaps))
        std_gap = float(np.std(gaps))
        val = std_gap / (mean_gap + 1e-12)
    return DescriptiveResult(
        name="svsph",
        value=float(val),
        extra={"n": n, "method": method},
    )


spat = spatial_phase


def cheatsheet() -> str:
    return "spatial_phase({}) -> Spatial phase transition (chaos/order)"

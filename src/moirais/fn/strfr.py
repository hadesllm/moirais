"""Stellar luminosity function. 'I am not from your planet.' -- Starfire"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def luminosity_function(
    magnitudes: np.ndarray | list[float],
    *,
    bin_width: float = 0.5,
    volume: float = 1.0,
) -> DescriptiveResult:
    """Compute the luminosity function (number density per magnitude bin).

    Estimates Phi(M) = N(M) / (V * delta_M) from a sample of absolute
    magnitudes, providing the space density of objects as a function of
    luminosity.

    Parameters
    ----------
    magnitudes : array
        Absolute magnitudes.
    bin_width : float
        Bin width in magnitudes.
    volume : float
        Survey volume (arbitrary units).

    Returns
    -------
    DescriptiveResult
        ``value`` = dict with ``'bin_centers'`` and ``'phi'`` (number density).
    """
    m = np.asarray(magnitudes, dtype=float).ravel()
    m = m[np.isfinite(m)]
    if len(m) < 3:
        raise ValueError("Need at least 3 magnitudes")
    if volume <= 0:
        raise ValueError("volume must be positive")
    bins = np.arange(m.min(), m.max() + bin_width, bin_width)
    counts, edges = np.histogram(m, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    phi = counts / (volume * bin_width)
    log_phi = np.where(phi > 0, np.log10(phi), np.nan)
    return DescriptiveResult(
        name="Luminosity function",
        value={"bin_centers": centers.tolist(), "phi": phi.tolist()},
        extra={
            "n": len(m),
            "bin_width": bin_width,
            "volume": volume,
            "log_phi": [float(v) if np.isfinite(v) else None for v in log_phi],
            "mag_range": [float(m.min()), float(m.max())],
            "total_density": float(np.sum(phi) * bin_width),
        },
    )


strfr = luminosity_function


def cheatsheet() -> str:
    return "luminosity_function({}) -> Stellar luminosity function. 'I am not from your planet.' --"

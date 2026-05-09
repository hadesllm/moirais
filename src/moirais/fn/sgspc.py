"""Spectral density estimation for spatial data."""

from __future__ import annotations

from ._containers import DescriptiveResult


def spectral_density(Z, coords, n_freq=50):
    """Estimate spectral density of a spatial process on a grid.

    Uses 2D FFT on gridded data (nearest-neighbor interpolation if irregular).

    .. epigraph:: "You spoony bard!" -- Tellah, Final Fantasy IV

    Parameters
    ----------
    Z : array_like
        Observed values.
    coords : array_like
        Coordinates, shape ``(n, 2)``.
    n_freq : int
        Grid resolution for FFT.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np

    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    xi = np.linspace(xmin, xmax, n_freq)
    yi = np.linspace(ymin, ymax, n_freq)
    grid = np.full((n_freq, n_freq), np.nan)

    for k in range(len(Z)):
        ix = int((coords[k, 0] - xmin) / (xmax - xmin + 1e-10) * (n_freq - 1))
        iy = int((coords[k, 1] - ymin) / (ymax - ymin + 1e-10) * (n_freq - 1))
        ix = min(ix, n_freq - 1)
        iy = min(iy, n_freq - 1)
        grid[iy, ix] = Z[k]

    grid = np.where(np.isnan(grid), np.nanmean(grid), grid)
    grid -= grid.mean()

    fft2 = np.fft.fft2(grid)
    power = np.abs(fft2) ** 2 / (n_freq**2)
    power_shifted = np.fft.fftshift(power)

    freqs = np.fft.fftshift(np.fft.fftfreq(n_freq))
    radial_freq = np.sqrt(freqs[:, None] ** 2 + freqs[None, :] ** 2)
    freq_bins = np.linspace(0, radial_freq.max(), n_freq // 2)
    radial_power = np.zeros(len(freq_bins) - 1)
    for i in range(len(freq_bins) - 1):
        mask = (radial_freq >= freq_bins[i]) & (radial_freq < freq_bins[i + 1])
        if mask.any():
            radial_power[i] = power_shifted[mask].mean()

    mids = 0.5 * (freq_bins[:-1] + freq_bins[1:])

    return DescriptiveResult(
        name="spectral_density",
        value=float(radial_power.max()),
        extra={
            "frequencies": mids.tolist(),
            "power": radial_power.tolist(),
            "peak_frequency": float(mids[np.argmax(radial_power)]) if len(radial_power) > 0 else 0.0,
        },
    )


sgspc = spectral_density


def cheatsheet() -> str:
    return "spectral_density({}) -> Spectral density estimation for spatial data."

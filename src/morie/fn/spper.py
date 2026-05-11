"""Spatial periodogram (Schabenberger & Gotway Ch 6)."""

import numpy as np


def spper(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    n_freq: int = 20,
) -> dict:
    """
    Compute the spatial periodogram (spectral density estimate).

    Evaluates the periodogram at a grid of spatial frequencies to
    detect periodic patterns in the spatial data.

    :param coords: Observation coordinates (n, 2).
    :param values: Observed values (n,).
    :param n_freq: Number of frequency bins per axis.
    :return: dict with ``freq_x``, ``freq_y``, ``power`` (n_freq x n_freq),
        ``total_power``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 6.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = int(values) if values.ndim == 0 else len(values)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")

    z = values - values.mean()

    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    if x_range == 0:
        x_range = 1.0
    if y_range == 0:
        y_range = 1.0

    freq_x = np.linspace(-np.pi / x_range, np.pi / x_range, n_freq)
    freq_y = np.linspace(-np.pi / y_range, np.pi / y_range, n_freq)

    power = np.zeros((n_freq, n_freq))
    for i, fx in enumerate(freq_x):
        for j, fy in enumerate(freq_y):
            phase = coords[:, 0] * fx + coords[:, 1] * fy
            real_part = np.sum(z * np.cos(phase))
            imag_part = np.sum(z * np.sin(phase))
            power[i, j] = (real_part ** 2 + imag_part ** 2) / n

    return {
        "freq_x": freq_x,
        "freq_y": freq_y,
        "power": power,
        "total_power": float(power.sum()),
        "n": n,
        "n_freq": n_freq,
    }


spper_fn = spper


def cheatsheet() -> str:
    return "spper({}) -> Spatial periodogram (spectral density)."

# morie.fn — function file (hadesllm/morie)
"""1-D curvelet-like transform via wrapping."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You were the chosen one!"


def curvelet(x, n_scales: int = 4, **kwargs) -> DescriptiveResult:
    """1-D curvelet-like transform via frequency-domain windowing.

    Partitions the frequency domain into dyadic scales with smooth
    windows, analogous to a simplified 1-D curvelet decomposition.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    n_scales : int
        Number of decomposition scales (default 4).

    Returns
    -------
    DescriptiveResult
        ``value`` is number of scales; ``extra`` has ``coefficients``
        (list of arrays per scale), ``energies`` (per-scale energy).

    References
    ----------
    Candes, E. J., & Donoho, D. L. (2004). New tight frames of curvelets
    and optimal representations of objects with piecewise C^2
    singularities. *Comm. Pure Appl. Math.*, 57(2), 219-266.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    X = np.fft.fft(x)
    freqs = np.abs(np.fft.fftfreq(N, d=1.0 / N))

    coefficients = []
    energies = []
    remaining = X.copy()

    for s in range(n_scales):
        if s == n_scales - 1:
            c = np.fft.ifft(remaining)
            coefficients.append(np.real(c))
            energies.append(float(np.sum(np.abs(remaining) ** 2)))
        else:
            lo = N / (2 ** (n_scales - s))
            hi = N / (2 ** (n_scales - s - 1))
            window = np.zeros(N)
            for k in range(N):
                f = freqs[k]
                if lo <= f < hi:
                    t = (f - lo) / (hi - lo + 1e-15)
                    window[k] = 0.5 * (1.0 + np.cos(np.pi * (1.0 - t)))
                elif f < lo:
                    window[k] = 0.0
            band = X * window
            coefficients.append(np.real(np.fft.ifft(band)))
            energies.append(float(np.sum(np.abs(band) ** 2)))
            remaining = remaining * (1.0 - window)

    return DescriptiveResult(
        name="curvelet",
        value=n_scales,
        extra={"coefficients": coefficients, "energies": energies},
    )


curlt = curvelet


def cheatsheet() -> str:
    return "curvelet({}) -> 1-D curvelet-like transform via wrapping."

# morie.fn — function file (hadesllm/morie)
"""Mastering others is strength; mastering yourself is true power. — Lao Tzu"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def greens_convolve(
    signal: np.ndarray,
    *,
    kernel: str = "gaussian",
    sigma: float = 1.0,
    n_kernel: int | None = None,
) -> DescriptiveResult:
    """Convolve a 1-D signal with a Green's function kernel.

    Green's functions are fundamental solutions to linear differential
    operators.  This function provides convolution with common Green's
    function kernels for signal smoothing and system response analysis.

    Parameters
    ----------
    signal : array
        1-D input signal.
    kernel : str
        ``'gaussian'``, ``'exponential'``, ``'lorentzian'``, or ``'heat'``.
    sigma : float
        Scale parameter of the kernel.
    n_kernel : int or None
        Kernel half-width in samples.  Default ``4 * ceil(sigma)``.

    Returns
    -------
    DescriptiveResult
        ``value`` = convolved signal as list.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    if len(signal) < 2:
        raise ValueError("Signal must have at least 2 samples")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if n_kernel is None:
        n_kernel = int(np.ceil(sigma) * 4)
    n_kernel = max(n_kernel, 1)
    t = np.arange(-n_kernel, n_kernel + 1, dtype=float)
    if kernel == "gaussian":
        g = np.exp(-(t**2) / (2 * sigma**2))
    elif kernel == "exponential":
        g = np.exp(-np.abs(t) / sigma)
    elif kernel == "lorentzian":
        g = sigma / (t**2 + sigma**2)
    elif kernel == "heat":
        g = np.exp(-(t**2) / (4 * sigma)) / np.sqrt(4 * np.pi * sigma)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    g /= np.sum(g)
    convolved = np.convolve(signal, g, mode="same")
    return DescriptiveResult(
        name=f"Green's function convolution ({kernel})",
        value=convolved.tolist(),
        extra={
            "kernel": kernel,
            "sigma": sigma,
            "kernel_length": len(g),
            "signal_length": len(signal),
            "snr_improvement_db": float(10 * np.log10(np.var(signal) / max(np.var(signal - convolved), 1e-30))),
        },
    )


grnln = greens_convolve


def cheatsheet() -> str:
    return "greens_convolve({}) -> Green's function convolution. 'In brightest day, in blackest"

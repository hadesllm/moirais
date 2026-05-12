"""The happiness of your life depends upon the quality of your thoughts. -- Marcus Aurelius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def window_function(
    n: int,
    *,
    kind: str = "hann",
) -> DescriptiveResult:
    """Generate a spectral window function of length *n*.

    Parameters
    ----------
    n : int
        Window length (number of samples).
    kind : str
        Window type: ``'hann'``, ``'hamming'``, ``'blackman'``, ``'bartlett'``,
        ``'kaiser'`` (beta=14), ``'rectangular'``.

    Returns
    -------
    DescriptiveResult
        ``value`` = dict with ``'window'`` array and ``'coherent_gain'``.
    """
    if n < 1:
        raise ValueError("Window length must be >= 1")
    windows = {
        "hann": np.hanning,
        "hamming": np.hamming,
        "blackman": np.blackman,
        "bartlett": np.bartlett,
    }
    if kind == "rectangular":
        w = np.ones(n)
    elif kind == "kaiser":
        w = np.kaiser(n, beta=14.0)
    elif kind in windows:
        w = windows[kind](n)
    else:
        raise ValueError(f"Unknown window: {kind}. Use hann/hamming/blackman/bartlett/kaiser/rectangular.")
    coherent_gain = float(np.sum(w)) / n
    noise_bw = float(n * np.sum(w**2) / np.sum(w) ** 2)
    return DescriptiveResult(
        name=f"Window function ({kind})",
        value={"window": w.tolist(), "coherent_gain": coherent_gain},
        extra={
            "kind": kind,
            "n": n,
            "coherent_gain": coherent_gain,
            "noise_bandwidth": noise_bw,
            "max": float(np.max(w)),
            "min": float(np.min(w)),
        },
    )


wndwm = window_function


def cheatsheet() -> str:
    return "window_function({}) -> Window functions. 'I will fight for those who cannot fight f"

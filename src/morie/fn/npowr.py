# morie.fn -- function file (hadesllm/morie)
"""Noise power estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Impressive. Most impressive."


def noise_power(x, signal=None, **kwargs) -> DescriptiveResult:
    """Estimate the noise power in *x*.

    If *signal* is provided, noise is ``x - signal``. Otherwise the
    variance of *x* is used as a proxy.

    Parameters
    ----------
    x : array-like
        Observed (possibly noisy) signal.
    signal : array-like or None
        Clean reference signal. If None, var(x) is returned.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    if signal is not None:
        signal = np.asarray(signal, dtype=float)
        noise = x - signal
        pn = float(np.mean(noise**2))
    else:
        pn = float(np.var(x, ddof=0))
    return DescriptiveResult(
        name="noise_power",
        value=pn,
        extra={"noise_power": pn, "n": len(x)},
    )


npowr = noise_power


def cheatsheet() -> str:
    return "noise_power({}) -> Noise power estimation."

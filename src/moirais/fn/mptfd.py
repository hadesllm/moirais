# moirais.fn — function file (hadesllm/moirais)
"""Matching Pursuit Time-Frequency Distribution."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def mptfd_fn(x: np.ndarray, n_atoms: int = 50, fs: float = 1.0) -> DescriptiveResult:
    """Compute time-frequency distribution via Matching Pursuit.

    :param x: 1-D input signal.
    :param n_atoms: Maximum atoms to extract (default 50).
    :param fs: Sampling frequency (default 1.0).
    :return: DescriptiveResult with frequency bin count and TFD/time/freq arrays.
    """
    from moirais._decompose import mp_time_frequency

    x = np.asarray(x, dtype=float).ravel()
    tfd, t, f = mp_time_frequency(x, n_atoms=n_atoms, fs=fs)
    return DescriptiveResult(
        name="mp_tfd",
        value=tfd.shape[0],
        extra={"tfd": tfd, "time": t, "frequencies": f},
    )


mptfd = mptfd_fn


def cheatsheet() -> str:
    return "mptfd_fn({}) -> Matching Pursuit Time-Frequency Distribution."

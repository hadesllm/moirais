"""SMUAP point process EMG simulation."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def smuap_point_process_fn(
    n_mus: int = 10,
    firing_rates: np.ndarray | None = None,
    fs: float = 1000.0,
    duration: float = 1.0,
) -> SignalResult:
    """Simulate EMG via superposition of motor unit action potentials.

    :param n_mus: Number of motor units (default 10).
    :param firing_rates: Per-MU firing rates in Hz; random 5-30 if None.
    :param fs: Sampling frequency in Hz (default 1000).
    :param duration: Signal duration in seconds (default 1.0).
    :return: SignalResult with simulated EMG and event markers.
    """
    from moirais._biomodel import smuap_point_process

    emg, events = smuap_point_process(
        n_mus=n_mus,
        firing_rates=firing_rates,
        fs=fs,
        duration=duration,
    )
    return SignalResult(
        name="smuap_point_process",
        filtered=emg,
        fs=fs,
        n_samples=len(emg),
        extra={"events": events, "n_mus": n_mus, "firing_rates": firing_rates},
    )


smuap = smuap_point_process_fn


def cheatsheet() -> str:
    return "smuap_point_process_fn({}) -> SMUAP point process EMG simulation."

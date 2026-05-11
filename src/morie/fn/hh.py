# morie.fn — function file (hadesllm/morie)
"""Hodgkin-Huxley neuron model simulation."""

from __future__ import annotations

from ._containers import SignalResult


def hodgkin_huxley_fn(
    duration: float = 50.0,
    dt: float = 0.01,
    I_ext: float = 10.0,
) -> SignalResult:
    """Simulate Hodgkin-Huxley neuron membrane potential.

    :param duration: Simulation duration in ms (default 50).
    :param dt: Time step in ms (default 0.01).
    :param I_ext: External current in uA/cm^2 (default 10).
    :return: SignalResult with membrane voltage trace.
    """
    from morie._biomodel import hodgkin_huxley

    t, V = hodgkin_huxley(duration=duration, dt=dt, I_ext=I_ext)
    return SignalResult(
        name="hodgkin_huxley",
        filtered=V,
        fs=1.0 / dt,
        n_samples=len(V),
        extra={"time": t, "voltage": V, "duration": duration, "I_ext": I_ext},
    )


hh = hodgkin_huxley_fn


def cheatsheet() -> str:
    return "hodgkin_huxley_fn({}) -> Hodgkin-Huxley neuron model simulation."

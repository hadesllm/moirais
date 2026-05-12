# morie.fn -- function file (hadesllm/morie)
"""Phase-locked loop bandwidth."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I have a bad feeling about this."


def pll_bandwidth(loop_gain, natural_freq, **kwargs) -> DescriptiveResult:
    """Compute phase-locked loop noise bandwidth.

    For a second-order PLL with damping ratio zeta=0.707:
    B_n = (natural_freq / 2) * (zeta + 1/(4*zeta))

    Parameters
    ----------
    loop_gain : float
        Loop gain K.
    natural_freq : float
        Natural frequency omega_n in rad/s.

    Returns
    -------
    DescriptiveResult
    """
    if natural_freq <= 0:
        raise ValueError("Natural frequency must be positive.")
    zeta = kwargs.get("damping", 0.707)
    bn = (natural_freq / 2.0) * (zeta + 1.0 / (4.0 * zeta))
    bn_hz = bn / (2.0 * np.pi)
    return DescriptiveResult(
        name="pll_bandwidth",
        value=float(bn_hz),
        extra={
            "loop_gain": loop_gain,
            "natural_freq": natural_freq,
            "damping": zeta,
            "noise_bw_rad": float(bn),
            "noise_bw_hz": float(bn_hz),
        },
    )


pllbd = pll_bandwidth


def cheatsheet() -> str:
    return "pll_bandwidth({}) -> Phase-locked loop bandwidth."

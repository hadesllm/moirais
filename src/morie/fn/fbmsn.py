# morie.fn — function file (hadesllm/morie)
"""Fractional Brownian motion synthesis."""

from __future__ import annotations

from ._containers import SignalResult


def fbm_synthesis(N: int, H: float = 0.5) -> SignalResult:
    """Synthesize a fractional Brownian motion signal.

    'Truly wonderful the mind of a child is.'
    """
    from morie._spectral import fbm_synthesis as _backend

    fbm = _backend(N, H=H)
    return SignalResult(
        name="fbm_synthesis",
        filtered=fbm,
        fs=1.0,
        n_samples=len(fbm),
        extra={"H": H, "N": N},
    )


fbmsn = fbm_synthesis


def cheatsheet() -> str:
    return "fbm_synthesis({}) -> Fractional Brownian motion synthesis."

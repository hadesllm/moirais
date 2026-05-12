# morie.fn -- function file (hadesllm/morie)
"""Hilbert transform amplitude envelope."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def hilbert_envelope_fn(x: np.ndarray) -> SignalResult:
    """Compute the Hilbert transform amplitude envelope.

    'He who is brave is free. -- Seneca' -- Chirrut Imwe
    """
    from morie._detection import hilbert_envelope as _backend

    result = _backend(x)
    return SignalResult(
        name="hilbert_envelope",
        filtered=result,
        n_samples=int(len(x)),
    )


alias = hilbert_envelope_fn


def cheatsheet() -> str:
    return "hilbert_envelope_fn({}) -> Hilbert transform amplitude envelope."

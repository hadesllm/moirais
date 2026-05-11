# morie.fn — function file (hadesllm/morie)
"""Channel equalization (inverse filter)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You were the chosen one!"


def equalization_inverse(H, regularize=0.01, **kwargs) -> DescriptiveResult:
    """Compute a regularised inverse filter for channel equalisation.

    H_inv = conj(H) / (|H|^2 + eps)

    Parameters
    ----------
    H : array-like
        Channel frequency response.
    regularize : float
        Regularisation parameter (default 0.01).

    Returns
    -------
    DescriptiveResult
    """
    H = np.asarray(H, dtype=complex)
    H_inv = np.conj(H) / (np.abs(H) ** 2 + regularize)
    gain = float(np.mean(np.abs(H_inv)))
    return DescriptiveResult(
        name="equalization_inverse",
        value=gain,
        extra={
            "mean_gain": gain,
            "max_gain": float(np.max(np.abs(H_inv))),
            "regularize": regularize,
            "n_taps": len(H),
        },
    )


eqint = equalization_inverse


def cheatsheet() -> str:
    return "equalization_inverse({}) -> Channel equalization (inverse filter)."

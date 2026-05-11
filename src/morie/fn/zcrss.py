"""Zero-crossing detection. 'Your focus determines your reality.' -- Qui-Gon Jinn"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def zero_crossings(
    signal: np.ndarray,
    threshold: float = 0.0,
) -> DescriptiveResult:
    """Count and locate zero crossings in a signal.

    A zero crossing occurs when the signal changes sign between
    consecutive samples (relative to *threshold*).

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Input signal.
    threshold : float
        Reference level for crossing detection.

    Returns
    -------
    DescriptiveResult
        name='Zero Crossings', value=number of crossings,
        extra has 'indices' (sample indices where crossings occur),
        'rate' (crossings per sample), 'n_samples'.

    References
    ----------
    Kedem, B. (1986). Spectral analysis and discrimination by
    zero-crossings. *Proceedings of the IEEE*, 74(11), 1477-1493.
    doi:10.1109/PROC.1986.13663
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    N = len(x)

    if N < 2:
        return DescriptiveResult(
            name="Zero Crossings",
            value=0,
            extra={"indices": np.array([], dtype=int), "rate": 0.0, "n_samples": N},
        )

    shifted = x - threshold
    sign_changes = np.where(np.diff(np.sign(shifted)))[0]

    n_crossings = len(sign_changes)
    rate = n_crossings / (N - 1) if N > 1 else 0.0

    return DescriptiveResult(
        name="Zero Crossings",
        value=n_crossings,
        extra={
            "indices": sign_changes,
            "rate": float(rate),
            "n_samples": N,
            "threshold": threshold,
        },
    )


def cheatsheet() -> str:
    return "zero_crossings({}) -> Zero-crossing detection. 'Your focus determines your reality"

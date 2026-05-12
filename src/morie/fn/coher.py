# morie.fn — function file (hadesllm/morie)
"""Coherence between two signals.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult
def coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    *,
    nperseg: int = 256,
) -> DescriptiveResult:
    r"""Magnitude-squared coherence between two signals.

    Measures the linear relationship between *x* and *y* at each
    frequency, normalized to [0, 1]:

    .. math::

        C_{xy}(f) = \\frac{|P_{xy}(f)|^2}{P_{xx}(f) \\, P_{yy}(f)}

    Uses Welch's method for cross- and auto-spectral estimation.

    Parameters
    ----------
    x, y : array-like
        1-D input signals (must be the same length).
    fs : float
        Sampling frequency in Hz (default 1.0).
    nperseg : int
        Segment length for Welch estimation (default 256).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``frequencies``, ``coherence``,
        ``cross_spectrum``.

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 6.
    """
    from scipy.signal import csd, welch

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    seg = min(nperseg, n)

    f, Pxx = welch(x, fs=fs, nperseg=seg)
    _, Pyy = welch(y, fs=fs, nperseg=seg)
    _, Pxy = csd(x, y, fs=fs, nperseg=seg)

    coh = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-20)

    return DescriptiveResult(
        name="coherence",
        value=float(np.mean(coh)),
        extra={
            "frequencies": f,
            "coherence": coh,
            "cross_spectrum": Pxy,
        },
    )


coher = coherence


def cheatsheet() -> str:
    return "coherence({}) -> Magnitude-squared coherence between two signals."

# morie.fn — function file (hadesllm/morie)
"""Matched filter for template detection in biomedical signals.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 3.
"""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "This is the template you are looking for. --"


def matched_filter(
    x: np.ndarray,
    template: np.ndarray,
    fs: float = 1.0,
) -> SignalResult:
    """Apply a matched filter to detect a known template in signal *x*.

    The matched filter maximizes the signal-to-noise ratio at the
    output when the noise is white and Gaussian:

    .. math::

        y(n) = \\sum_{k=0}^{M-1} h(k) \\, x(n-k), \\quad
        h(k) = s(M-1-k)

    where :math:`s` is the template (time-reversed for correlation).

    Parameters
    ----------
    x : array-like
        Input signal to search.
    template : array-like
        Known template / reference waveform.
    fs : float
        Sampling frequency in Hz (default 1.0).

    Returns
    -------
    SignalResult
        ``filtered`` contains the matched-filter output (correlation),
        ``extra`` has ``peak_index`` and ``peak_snr``.

    References
    ----------
    Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
    Analysis*, 3rd ed. IEEE/Wiley, Chapter 3.
    """
    x = np.asarray(x, dtype=float).ravel()
    template = np.asarray(template, dtype=float).ravel()

    h = template[::-1] / (np.linalg.norm(template) + 1e-12)

    output = np.correlate(x, template, mode="full")
    output = output[: len(x)]

    peak_idx = int(np.argmax(np.abs(output)))
    noise_est = np.std(output) + 1e-12
    peak_snr = float(np.abs(output[peak_idx]) / noise_est)

    return SignalResult(
        name="matched_filter",
        filtered=output,
        fs=fs,
        n_samples=len(output),
        extra={"peak_index": peak_idx, "peak_snr": peak_snr},
    )


mchfl = matched_filter


def cheatsheet() -> str:
    return "matched_filter({}) -> Matched filter for template detection."

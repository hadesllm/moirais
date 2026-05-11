# morie.fn — function file (hadesllm/morie)
"""Multitaper power spectral density (DPSS windows).

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 5.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['psdmt']

_QUOTE = "Many windows reveal the truth. -- Mace Windu"


def psdmt(
    x: np.ndarray,
    fs: float = 1.0,
    *,
    nw: float = 4.0,
    n_tapers: int | None = None,
    nfft: int | None = None,
) -> DescriptiveResult:
    """Multitaper PSD using Slepian (DPSS) windows.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    fs : float
        Sampling frequency in Hz.
    nw : float
        Time-bandwidth product (default 4.0).
    n_tapers : int or None
        Number of tapers (default 2*nw - 1).
    nfft : int or None
        FFT length (default len(x)).

    Returns
    -------
    DescriptiveResult
    """
    from scipy.signal.windows import dpss

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n_tapers is None:
        n_tapers = int(2 * nw - 1)
    if nfft is None:
        nfft = n

    tapers, concentrations = dpss(n, nw, Kmax=n_tapers, return_ratios=True)
    if tapers.ndim == 1:
        tapers = tapers[np.newaxis, :]
        concentrations = np.atleast_1d(concentrations)

    nfreqs = nfft // 2 + 1
    psd = np.zeros(nfreqs)
    for k in range(len(concentrations)):
        xk = tapers[k] * x
        Xk = np.fft.rfft(xk, n=nfft)
        psd += concentrations[k] * np.abs(Xk) ** 2
    psd /= (fs * np.sum(concentrations))

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    return DescriptiveResult(
        name="psdmt",
        value=float(np.trapezoid(psd, freqs)),
        extra={
            "frequencies": freqs,
            "psd": psd,
            "n_tapers": len(concentrations),
            "concentrations": concentrations,
        },
    )


def cheatsheet() -> str:
    return "psdmt({}) -> Multitaper PSD."

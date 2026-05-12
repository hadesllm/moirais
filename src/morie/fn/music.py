# morie.fn -- function file (hadesllm/morie)
"""MUSIC spectral estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Statistics is the grammar of science. -- Karl Pearson"


def music_spectrum_fn(
    x: np.ndarray,
    nsources: int = 2,
    order: int = 16,
    nfft: int = 512,
    fs: float = 1.0,
) -> DescriptiveResult:
    r"""MUSIC (Multiple Signal Classification) spectral estimation.

    Estimates the pseudospectrum by projecting steering vectors onto
    the noise subspace of the autocorrelation matrix.

    .. math::

        P_{\\text{MUSIC}}(f) = \\frac{1}{\\mathbf{e}^H(f) E_n E_n^H \\mathbf{e}(f)}

    :param x: 1-D input signal.
    :param nsources: Number of signal sources (default 2).
    :param order: Correlation matrix dimension (default 16).
    :param nfft: Number of frequency bins (default 512).
    :param fs: Sampling frequency in Hz (default 1.0).
    :return: DescriptiveResult with MUSIC pseudospectrum.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    x_c = x - np.mean(x)
    R = np.zeros((order, order), dtype=complex)
    for i in range(order, n):
        seg = x_c[i - order : i][::-1]
        R += np.outer(seg, seg.conj())
    R /= n - order
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)
    noise_vecs = eigenvectors[:, idx[: order - nsources]]
    freqs = np.linspace(0, fs / 2, nfft)
    psd = np.zeros(nfft)
    for i, f in enumerate(freqs):
        w = 2 * np.pi * f / fs
        e = np.exp(-1j * w * np.arange(order))
        proj = noise_vecs.conj().T @ e
        denom = float(np.real(np.dot(proj.conj(), proj)))
        psd[i] = 1.0 / max(denom, 1e-20)
    return DescriptiveResult(
        name="music_spectrum",
        value=None,
        extra={"frequencies": freqs, "psd": psd, "nsources": nsources, "fs": fs},
    )


music = music_spectrum_fn


def cheatsheet() -> str:
    return "music_spectrum_fn({}) -> MUSIC spectral estimation."

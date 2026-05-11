# morie.fn — function file (hadesllm/morie)
"""Morphological Component Analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Life is really simple, but we insist on making it complicated. — Confucius"


def morphological_ca(x, n_iter: int = 100, **kwargs) -> DescriptiveResult:
    """Morphological component analysis: separate smooth and transient.

    Decomposes signal *x* into a smooth (low-frequency) component via
    iterative thresholding on the DCT domain and a transient (detail)
    component as the residual.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    n_iter : int
        Number of thresholding iterations (default 100).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``smooth``, ``transient``, ``n_iter``.

    References
    ----------
    Starck, J.-L., Elad, M., & Donoho, D. (2005). Image decomposition
    via the combination of sparse representations and a variational
    approach. *IEEE Trans. Image Process.*, 14(10), 1570-1582.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    from scipy.fft import dct, idct

    smooth = np.zeros(N)
    threshold = np.max(np.abs(dct(x, norm="ortho")))
    for i in range(n_iter):
        residual = x - smooth
        coeffs = dct(residual, norm="ortho")
        t = threshold * (1.0 - (i + 1) / n_iter)
        coeffs[np.abs(coeffs) < t] = 0.0
        smooth = smooth + idct(coeffs, norm="ortho")
    transient = x - smooth
    energy_smooth = float(np.sum(smooth**2))
    energy_total = float(np.sum(x**2)) + 1e-15
    return DescriptiveResult(
        name="morphological_ca",
        value=energy_smooth / energy_total,
        extra={"smooth": smooth, "transient": transient, "n_iter": n_iter},
    )


mca = morphological_ca


def cheatsheet() -> str:
    return "morphological_ca({}) -> Morphological Component Analysis."

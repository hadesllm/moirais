# morie.fn -- function file (hadesllm/morie)
"""Mexican hat (Ricker) wavelet."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I'm just a simple man trying to make my way in the universe."


def mexican_hat(N: int = 256) -> DescriptiveResult:
    r"""Mexican hat (Ricker) wavelet: negative normalized second derivative of Gaussian.

    .. math::

        \\psi(t) = \\frac{2}{\\sqrt{3}\\pi^{1/4}}
                   (1 - t^2) \\, e^{-t^2/2}

    Parameters
    ----------
    N : int
        Number of samples (default 256).

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``wavelet``, ``time``.
    """
    t = np.linspace(-5, 5, N)
    c = 2.0 / (np.sqrt(3) * np.pi**0.25)
    psi = c * (1 - t**2) * np.exp(-(t**2) / 2)
    return DescriptiveResult(
        name="mexican_hat",
        value=float(N),
        extra={"wavelet": psi, "time": t},
    )


mexht = mexican_hat


def cheatsheet() -> str:
    return "mexican_hat({}) -> Mexican hat (Ricker) wavelet."

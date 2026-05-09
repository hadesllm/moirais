"""Form factor."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You underestimate my power."


def form_factor(x, **kwargs) -> DescriptiveResult:
    """Compute the form factor of signal *x*.

    .. math::

        \\text{FF} = \\frac{\\text{RMS}(x)}{\\text{mean}(|x|)}

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    rms = float(np.sqrt(np.mean(x**2)))
    mean_abs = float(np.mean(np.abs(x)))
    ff = rms / mean_abs if mean_abs > 0 else float("inf")
    return DescriptiveResult(
        name="form_factor",
        value=ff,
        extra={"form_factor": ff, "rms": rms, "mean_abs": mean_abs, "n": len(x)},
    )


sfrmf = form_factor


def cheatsheet() -> str:
    return "form_factor({}) -> Form factor."

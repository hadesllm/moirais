# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Adaptive noise cancellation via LMS."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "Many of the truths we cling to depend on our point of view."


def anc_remove(signal, reference, mu: float = 0.01, order: int = 16) -> SignalResult:
    """Adaptive Noise Cancellation using LMS.

    Uses a reference noise signal to adaptively cancel noise from the
    primary signal.

    Parameters
    ----------
    signal : array-like
        Primary signal (signal + noise).
    reference : array-like
        Reference noise signal correlated with noise in primary.
    mu : float
        LMS step size. Default 0.01.
    order : int
        Filter order. Default 16.

    Returns
    -------
    SignalResult
    """
    from morie.fn.lmsfl import lms_filter

    sig = np.asarray(signal, dtype=float)
    ref = np.asarray(reference, dtype=float)
    result = lms_filter(ref, sig, mu=mu, order=order)
    cleaned = sig[: result.n_samples] - result.filtered
    return SignalResult(
        name="anc_remove",
        filtered=cleaned,
        fs=0.0,
        n_samples=len(cleaned),
        extra={"mu": mu, "order": order, "error": result.extra.get("error")},
    )


ancrm = anc_remove


def cheatsheet() -> str:
    return "anc_remove({}) -> Adaptive noise cancellation via LMS."

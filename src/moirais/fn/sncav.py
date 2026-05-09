"""Synchronized (trigger-based) average filter."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def sync_avg(x, trigger_indices, window: int = 100) -> SignalResult:
    """Compute a synchronized (trigger-locked) average of signal *x*.

    Parameters
    ----------
    x : array-like
        Input signal.
    trigger_indices : array-like of int
        Sample indices of trigger events.
    window : int
        Number of samples to include after each trigger. Default 100.

    Returns
    -------
    SignalResult
    """
    from moirais._filters import synchronized_average as _sa

    x = np.asarray(x, dtype=float)
    trigger_indices = np.asarray(trigger_indices, dtype=int)
    result = _sa(x, trigger_indices=trigger_indices, window=window)
    return SignalResult(name="sync_avg", filtered=result, fs=0.0, n_samples=result.shape[0])


sncav = sync_avg


def cheatsheet() -> str:
    return "sync_avg({}) -> Synchronized (trigger-based) average filter."

# moirais.fn — function file (hadesllm/moirais)
"""ECG multi-lead plot visualization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def plot_ecg_leads_fn(
    signals: np.ndarray,
    fs: float = 500.0,
    lead_names: list[str] | None = None,
    duration: float | None = None,
) -> DescriptiveResult:
    """Plot multi-lead ECG signals in a stacked subplot layout.

    Parameters
    ----------
    signals : np.ndarray
        2-D array of shape ``(n_leads, n_samples)`` or 1-D for a single lead.
    fs : float
        Sampling frequency in Hz (default 500).
    lead_names : list[str], optional
        Label for each lead.
    duration : float, optional
        Truncate display to this many seconds.

    Returns
    -------
    DescriptiveResult
        *value* is number of leads; *extra* contains ``figure``.

    References
    ----------
    Kligfield, P. et al. (2007). Recommendations for the standardization
        and interpretation of the electrocardiogram. *Circulation*, 115(10),
        1306--1324.
    """
    from moirais._bioplot import plot_ecg_leads

    signals = np.asarray(signals, dtype=float)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    n_leads = signals.shape[0]
    fig = plot_ecg_leads(signals, fs=fs, lead_names=lead_names, duration=duration)
    return DescriptiveResult(
        name="ecg_plot",
        value=n_leads,
        extra={"figure": fig, "n_leads": n_leads, "fs": fs},
    )


ecgplt = plot_ecg_leads_fn


def cheatsheet() -> str:
    return "plot_ecg_leads_fn({}) -> ECG multi-lead plot visualization."

# morie.fn -- function file (hadesllm/morie)
"""EEG montage plot visualization."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def plot_eeg_montage_fn(
    channels: np.ndarray,
    fs: float = 256.0,
    channel_names: list[str] | None = None,
    duration: float | None = None,
) -> DescriptiveResult:
    """Plot multi-channel EEG in a vertically stacked montage.

    Parameters
    ----------
    channels : np.ndarray
        2-D array ``(n_channels, n_samples)`` or 1-D for single channel.
    fs : float
        Sampling frequency in Hz (default 256).
    channel_names : list[str], optional
        Label for each channel.
    duration : float, optional
        Truncate display to this many seconds.

    Returns
    -------
    DescriptiveResult
        *value* is number of channels; *extra* contains ``figure``.

    References
    ----------
    Jasper, H. H. (1958). The ten-twenty electrode system of the
        International Federation. *Electroencephalography and Clinical
        Neurophysiology*, 10, 370--375.
    """
    from morie._bioplot import plot_eeg_montage

    channels = np.asarray(channels, dtype=float)
    if channels.ndim == 1:
        channels = channels.reshape(1, -1)
    n_ch = channels.shape[0]
    fig = plot_eeg_montage(channels, fs=fs, channel_names=channel_names, duration=duration)
    return DescriptiveResult(
        name="eeg_montage",
        value=n_ch,
        extra={"figure": fig, "n_channels": n_ch, "fs": fs},
    )


eegplt = plot_eeg_montage_fn


def cheatsheet() -> str:
    return "plot_eeg_montage_fn({}) -> EEG montage plot visualization."

# moirais.fn — function file (hadesllm/moirais)
"""EOG eye movement artifact detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "So this is how liberty dies... with thunderous applause."


def eog_detect(x, fs: float = 256.0, threshold: float | None = None, **kwargs) -> DescriptiveResult:
    """Detect EOG (electrooculography) eye movement artifacts.

    Uses amplitude thresholding on the derivative to find saccades/blinks.

    Parameters
    ----------
    x : array-like
        EOG signal.
    fs : float
        Sampling frequency in Hz.
    threshold : float or None
        Derivative threshold. If None, uses 3 * std(dx).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    dx = np.diff(x) * fs
    if threshold is None:
        threshold = 3.0 * np.std(dx) if len(dx) > 0 else 1.0
    artifacts = np.where(np.abs(dx) > threshold)[0]
    groups = []
    if len(artifacts) > 0:
        group_start = artifacts[0]
        for i in range(1, len(artifacts)):
            if artifacts[i] - artifacts[i - 1] > int(fs * 0.1):
                groups.append((int(group_start), int(artifacts[i - 1])))
                group_start = artifacts[i]
        groups.append((int(group_start), int(artifacts[-1])))
    return DescriptiveResult(
        name="eog_detect",
        value=float(len(groups)),
        extra={
            "artifact_indices": artifacts,
            "artifact_groups": groups,
            "derivative": dx,
            "threshold": threshold,
            "fs": fs,
        },
    )


eogdt = eog_detect


def cheatsheet() -> str:
    return "eog_detect({}) -> EOG eye movement artifact detection."

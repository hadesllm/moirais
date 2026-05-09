"""Time-Frequency Distribution feature extraction."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def tfdft_fn(tfd: np.ndarray, t: np.ndarray, f: np.ndarray) -> DescriptiveResult:
    """Extract statistical features from a time-frequency distribution.

    :param tfd: 2-D TFD array (freq x time).
    :param t: Time axis array.
    :param f: Frequency axis array.
    :return: DescriptiveResult with total energy and feature dict.
    """
    from moirais._decompose import tfd_features

    tfd = np.asarray(tfd, dtype=float)
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    features = tfd_features(tfd, t, f)
    return DescriptiveResult(
        name="tfd_features",
        value=features["total_energy"],
        extra=features,
    )


tfdft = tfdft_fn


def cheatsheet() -> str:
    return "tfdft_fn({}) -> Time-Frequency Distribution feature extraction."

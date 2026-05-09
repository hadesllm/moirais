# moirais.fn — function file (hadesllm/moirais)
"""Ensemble Empirical Mode Decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def eemd_fn(x: np.ndarray, n_ensembles: int = 100, noise_std: float = 0.2) -> DescriptiveResult:
    """Decompose signal via Ensemble EMD (noise-assisted).

    :param x: 1-D input signal.
    :param n_ensembles: Number of noise-added trials (default 100).
    :param noise_std: Noise amplitude as fraction of signal std (default 0.2).
    :return: DescriptiveResult with IMF count and averaged IMFs.
    """
    from moirais._decompose import ensemble_emd

    x = np.asarray(x, dtype=float).ravel()
    imfs = ensemble_emd(x, n_ensembles=n_ensembles, noise_std=noise_std)
    return DescriptiveResult(
        name="ensemble_emd",
        value=len(imfs),
        extra={"imfs": imfs, "n_ensembles": n_ensembles},
    )


eemd = eemd_fn


def cheatsheet() -> str:
    return "eemd_fn({}) -> Ensemble Empirical Mode Decomposition."

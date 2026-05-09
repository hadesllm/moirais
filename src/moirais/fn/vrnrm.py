"""Normalized variance reduction for spatial model fit."""

from __future__ import annotations

from ._containers import DescriptiveResult


def normalized_variance(Z, fitted) -> DescriptiveResult:
    """Compute normalized variance reduction: 1 - SS_res/SS_tot.

    :param Z: Original data matrix.
    :param fitted: Fitted/reconstructed data.
    :return: DescriptiveResult with variance reduction ratio.

    .. epigraph:: "I am atomic." -- Garou, One Punch Man
    """
    import numpy as np

    Z = np.asarray(Z, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    ss_tot = np.nansum((Z - np.nanmean(Z)) ** 2)
    ss_res = np.nansum((Z - fitted) ** 2)
    vr = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return DescriptiveResult(
        name="normalized_variance",
        value=vr,
        extra={"ss_total": float(ss_tot), "ss_residual": float(ss_res)},
    )


vrnrm = normalized_variance


def cheatsheet() -> str:
    return "normalized_variance({}) -> Normalized variance reduction for spatial model fit."

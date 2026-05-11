# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""A-M fit statistic (normalized variance reduction)."""

from __future__ import annotations

from ._containers import DescriptiveResult


def am_fit_statistic(Z, zhat, alpha, beta) -> DescriptiveResult:
    """Compute A-M normalized variance reduction (Eq 2.9 in Armstrong).

    :param Z: Respondent x stimulus perceptual data.
    :param zhat: Estimated stimulus positions.
    :param alpha: Per-respondent intercepts.
    :param beta: Per-respondent weights.
    :return: DescriptiveResult with fit statistic.

    .. epigraph:: "El Psy Kongroo." -- Rintaro Okabe, Steins;Gate
    """
    import numpy as np

    Z = np.asarray(Z, dtype=float)
    zhat = np.asarray(zhat, dtype=float).ravel()
    alpha = np.asarray(alpha, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    n_resp, n_stim = Z.shape
    fitted = alpha[:, None] + beta[:, None] * zhat[None, :]
    ss_res = np.nansum((Z - fitted) ** 2)
    ss_tot = np.nansum((Z - np.nanmean(Z)) ** 2)
    fit = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return DescriptiveResult(
        name="am_fit_statistic",
        value=float(fit),
        extra={"ss_residual": float(ss_res), "ss_total": float(ss_tot)},
    )


amfit = am_fit_statistic


def cheatsheet() -> str:
    return "am_fit_statistic({}) -> A-M fit statistic (normalized variance reduction)."

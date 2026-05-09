# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""A-M residual matrix computation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def am_residuals(Z, zhat, alpha, beta) -> DescriptiveResult:
    """Compute residuals from A-M model: e_ij = z_ij - (alpha_i + beta_i * zhat_j).

    :param Z: Respondent x stimulus perceptual data.
    :param zhat: Estimated stimulus positions.
    :param alpha: Per-respondent intercepts.
    :param beta: Per-respondent weights.
    :return: DescriptiveResult with residual matrix.

    .. epigraph:: "Shine, Kakyoin!" -- DIO, JoJo's Bizarre Adventure
    """
    import numpy as np

    Z = np.asarray(Z, dtype=float)
    zhat = np.asarray(zhat, dtype=float).ravel()
    alpha = np.asarray(alpha, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    fitted = alpha[:, None] + beta[:, None] * zhat[None, :]
    residuals = Z - fitted
    rmse = float(np.sqrt(np.nanmean(residuals**2)))
    return DescriptiveResult(
        name="am_residuals",
        value=rmse,
        extra={"residuals": residuals.tolist(), "rmse": rmse, "max_abs_resid": float(np.nanmax(np.abs(residuals)))},
    )


amres = am_residuals


def cheatsheet() -> str:
    return "am_residuals({}) -> A-M residual matrix computation."

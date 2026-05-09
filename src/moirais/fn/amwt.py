# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""A-M per-respondent weight estimation."""

from __future__ import annotations

from ._containers import DescriptiveResult


def am_weight_estimate(Z, zhat) -> DescriptiveResult:
    """Estimate per-respondent weights (beta_i) via OLS on stimulus positions.

    z_ij = alpha_i + beta_i * zhat_j + error

    :param Z: Respondent x stimulus perceptual data.
    :param zhat: Estimated stimulus positions (1-D).
    :return: DescriptiveResult with alpha and beta arrays.

    .. epigraph:: "Gear Fifth!" -- Monkey D. Luffy, One Piece
    """
    import numpy as np

    Z = np.asarray(Z, dtype=float)
    zhat = np.asarray(zhat, dtype=float).ravel()
    n_resp = Z.shape[0]
    alphas = np.zeros(n_resp)
    betas = np.zeros(n_resp)
    X = np.column_stack([np.ones_like(zhat), zhat])
    for i in range(n_resp):
        y = Z[i]
        mask = ~np.isnan(y)
        if mask.sum() >= 2:
            Xm = X[mask]
            ym = y[mask]
            coef = np.linalg.lstsq(Xm, ym, rcond=None)[0]
            alphas[i] = coef[0]
            betas[i] = coef[1]
    return DescriptiveResult(
        name="am_weight_estimate",
        value=float(np.mean(betas)),
        extra={"alphas": alphas.tolist(), "betas": betas.tolist(), "n_respondents": n_resp},
    )


amwt = am_weight_estimate


def cheatsheet() -> str:
    return "am_weight_estimate({}) -> A-M per-respondent weight estimation."

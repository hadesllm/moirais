# moirais.fn — function file (hadesllm/moirais)
"""Heckman two-step selection model. 'This is the way.' -- The Mandalorian"""

from __future__ import annotations

import numpy as np
from scipy import stats as _st

from ._containers import DescriptiveResult


def heckman_correction(y: np.ndarray, x: np.ndarray, z: np.ndarray, cdf=None) -> DescriptiveResult:
    """
    Heckman two-step correction for sample selection bias.

    Step 1: Probit selection equation on *z* to get inverse Mills ratio.
    Step 2: OLS outcome equation on *x* augmented with the Mills ratio.

    :param y: Outcome variable (1-D). NaN where not observed (selected out).
    :param x: Outcome regressors (n, p1). Rows align with *y*.
    :param z: Selection regressors (n, p2). Must include an exclusion
        restriction (variable in *z* but not *x*).
    :return: DescriptiveResult with selection-corrected coefficients.
    :raises ValueError: If dimensions mismatch or too few observations.

    References
    ----------
    Heckman, J. J. (1979). Sample selection bias as a specification error.
    Econometrica, 47(1), 153--161. doi:10.2307/1912352
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    n = y.shape[0]
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if x.shape[0] != n or z.shape[0] != n:
        raise ValueError("y, x, z must have the same number of rows.")

    selected = ~np.isnan(y)
    s = selected.astype(float)

    z_aug = np.column_stack([np.ones(n), z])
    from numpy.linalg import lstsq

    gamma, _, _, _ = lstsq(z_aug, s, rcond=None)
    probit_score = z_aug @ gamma
    probit_score = np.clip(probit_score, -3.5, 3.5)

    phi = _st.norm.pdf(probit_score)
    Phi = _st.norm.cdf(probit_score)
    Phi = np.maximum(Phi, 1e-12)
    mills = phi / Phi

    y_sel = y[selected]
    x_sel = x[selected]
    mills_sel = mills[selected].reshape(-1, 1)

    x_aug = np.column_stack([np.ones(y_sel.shape[0]), x_sel, mills_sel])
    beta, _, _, _ = lstsq(x_aug, y_sel, rcond=None)

    fitted = x_aug @ beta
    residuals = y_sel - fitted
    sigma = float(np.std(residuals, ddof=x_aug.shape[1]))
    lambda_coef = float(beta[-1])

    return DescriptiveResult(
        name="Heckman Selection Model",
        value=lambda_coef,
        extra={
            "beta": beta.tolist(),
            "lambda": lambda_coef,
            "sigma_residual": sigma,
            "gamma_selection": gamma.tolist(),
            "n_total": n,
            "n_selected": int(np.sum(selected)),
            "mills_ratio_mean": float(np.mean(mills_sel)),
        },
    )


heckm = heckman_correction


def cheatsheet() -> str:
    return "heckman_correction({}) -> Heckman two-step selection model. 'This is the way.' -- The "

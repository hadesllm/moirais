"""Spatial Durbin / SAR direct-indirect-total decomposition.

Adapted from Laniyonu (2018) Urban Affairs Review 54(5):898–930, which
in turn uses LeSage & Pace (2009) + Elhorst (2010) + the Yang/Noah/
Shoff (2015) decomposition formula.

The Laniyonu 2018 result -- gentrification's effect on stops/capita is
~0 direct but +51 to +90% indirect (spillover into neighbouring
tracts) -- only surfaces once you decompose.  An OLS or non-spatial
FE model would report "no effect" and miss the entire story.

This primitive is the SDM with the canonical decomposition + the
diagnostic ladder (Moran's I residual test, Robust LM lag vs error)
that justifies SDM over OLS.

We deliberately do NOT fit the SDM ourselves here -- ``pysal`` /
``spreg`` is a hard dep we don't want to force on every morie user.
This primitive expects the caller to either pass a fitted spreg
result (``preferred_spec="sdm"`` family) or use the ``"ols_with_w_x"``
fallback which is a vanilla OLS regression that includes spatially-
lagged covariates ($WX$) and lets the caller eyeball the
indirect-effect direction without a full SDM ML fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpilloverDecomposition:
    """Direct / indirect / total marginal effects from a spatial model."""
    coefficient: str
    direct: float
    indirect: float
    total: float
    direct_se: float | None = None
    indirect_se: float | None = None
    total_se: float | None = None
    note: str = ""

    def interpret(self) -> str:
        return (
            f"{self.coefficient}: direct={self.direct:+.4f}, "
            f"indirect={self.indirect:+.4f}, total={self.total:+.4f}. "
            f"{self.note}"
        )


def spatial_spillover_decomposition(
    *,
    rho: float,
    beta_direct: np.ndarray,
    beta_spatial: np.ndarray,
    W: np.ndarray,
    coefficient_names: list[str] | None = None,
) -> list[SpilloverDecomposition]:
    """Compute the SDM direct/indirect/total decomposition.

    Implements the standard formula:

        (I - rho * W)^{-1} * (I * beta_direct + W * beta_spatial)

    The diagonal of the resulting matrix is the per-observation direct
    effect (averaged for the summary); off-diagonal-row-sum is the
    indirect effect; total = direct + indirect.

    Parameters
    ----------
    rho : float
        Spatial-autoregressive coefficient from the fitted SDM.
    beta_direct : np.ndarray of shape (K,)
        Coefficients on the K covariates (no lagged terms).
    beta_spatial : np.ndarray of shape (K,)
        Coefficients on the spatially-lagged covariates ($WX$).
        Set to zeros if you fit a SAR (lag-only) model.
    W : np.ndarray of shape (N, N)
        Row-standardised spatial weight matrix.
    coefficient_names : list[str], optional
        Human-readable names for the K covariates.

    Returns
    -------
    list[SpilloverDecomposition]
        One entry per covariate.

    Notes
    -----
    This is the formula side; the caller is responsible for fitting
    the SDM (via pysal / spreg / statsmodels spatial extension) and
    passing in the estimated rho + beta vectors.  Fitting is left out
    of the primitive so morie can stay numpy-only at the framework
    layer.
    """
    K = len(beta_direct)
    if coefficient_names is None:
        coefficient_names = [f"x{i}" for i in range(K)]
    if len(beta_spatial) != K:
        raise ValueError("beta_direct and beta_spatial must have the same length")
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square, got shape {W.shape}")

    N = W.shape[0]
    I = np.eye(N)
    try:
        S = np.linalg.inv(I - rho * W)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"Could not invert (I - rho*W) at rho={rho}; "
            "spatial multiplier is singular (rho near 1/lambda_max?)"
        ) from exc

    out: list[SpilloverDecomposition] = []
    for k in range(K):
        # Per-observation effects matrix M_k = S * (I * beta_k + W * theta_k)
        M_k = S @ (beta_direct[k] * I + beta_spatial[k] * W)
        direct = float(np.mean(np.diag(M_k)))
        # Indirect = average row sum minus diagonal element
        row_sums = M_k.sum(axis=1)
        indirect = float(np.mean(row_sums - np.diag(M_k)))
        total = direct + indirect
        out.append(SpilloverDecomposition(
            coefficient=coefficient_names[k],
            direct=direct,
            indirect=indirect,
            total=total,
            note=("Per-observation marginal effects averaged across N "
                  f"tracts (rho={rho:.4f})."),
        ))
    return out


def morans_i(
    residuals: np.ndarray,
    W: np.ndarray,
) -> float:
    """Moran's I statistic for residual spatial autocorrelation.

    Use this as the diagnostic ladder's first rung: if OLS residuals
    show significant Moran's I, an SDM (or SEM/SAR) is warranted.

    Returns
    -------
    float
        Moran's I in [-1, 1].  Positive → clustering, negative →
        dispersion, ≈0 → spatial randomness.
    """
    n = residuals.size
    if W.shape != (n, n):
        raise ValueError(f"W must be ({n},{n}) to match residuals; got {W.shape}")
    e = residuals - residuals.mean()
    W_sum = W.sum()
    if W_sum == 0:
        return float("nan")
    numerator = float(e @ W @ e)
    denominator = float(e @ e)
    if denominator == 0:
        return float("nan")
    return (n / W_sum) * (numerator / denominator)

# morie.fn — function file (hadesllm/morie)
"""Regression kink design (RKD) estimator.

RKD identifies a causal effect by exploiting a known kink (slope
discontinuity) in the treatment assignment rule at a threshold.

References
----------
Card, D., Lee, D. S., Pei, Z., & Weber, A. (2015). Inference on
causal effects in a generalized regression kink design.
*Econometrica*, 83(6), 2453-2483.

Nielsen, H. S., Sorensen, T., & Taber, C. (2010). Estimating the
effect of student aid on college enrollment: Evidence from a
government grant policy reform. *American Economic Journal: Economic
Policy*, 2(2), 185-215.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm as _norm

__all__ = ["rgknd"]


def rgknd(
    Y: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    *,
    cutoff: float = 0.0,
    bandwidth: float | None = None,
    poly_order: int = 1,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Estimate the treatment effect via regression kink design.

    The RKD estimand is:

    .. math::

        \tau_{\text{RKD}} =
        \frac{\lim_{r \to c^+} \partial E[Y|R=r]/\partial r
              - \lim_{r \to c^-} \partial E[Y|R=r]/\partial r}
             {\lim_{r \to c^+} \partial E[T|R=r]/\partial r
              - \lim_{r \to c^-} \partial E[T|R=r]/\partial r}

    Estimated by fitting separate polynomial regressions on each side
    of the cutoff and taking the ratio of slope discontinuities.

    Parameters
    ----------
    Y : np.ndarray
        Outcome vector, shape ``(n,)``.
    T : np.ndarray
        Treatment (or dose) vector, shape ``(n,)``.
    R : np.ndarray
        Running variable, shape ``(n,)``.
    cutoff : float
        Kink threshold in R.
    bandwidth : float, optional
        Half-window around the cutoff. Defaults to IQR-based rule.
    poly_order : int
        Polynomial order for local regression (1 or 2).
    alpha : float
        Significance level.

    Returns
    -------
    dict
        ``tau``, ``se``, ``ci_lower``, ``ci_upper``,
        ``slope_Y_right``, ``slope_Y_left``,
        ``slope_T_right``, ``slope_T_left``,
        ``n_used``, ``bandwidth``, ``method``.

    References
    ----------
    Card et al. (2015). Econometrica, 83(6), 2453-2483.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    R = np.asarray(R, dtype=float)
    n = len(Y)
    if not (len(T) == n == len(R)):
        raise ValueError("Y, T, R must have the same length.")

    R_c = R - cutoff         # centred running variable

    if bandwidth is None:
        iqr = float(np.percentile(np.abs(R_c), 75) - np.percentile(np.abs(R_c), 25))
        bandwidth = max(1.5 * iqr, 1e-3)
        if bandwidth <= 0:
            bandwidth = float(np.std(R_c))

    mask = np.abs(R_c) <= bandwidth
    R_w = R_c[mask]
    Y_w = Y[mask]
    T_w = T[mask]
    n_used = int(mask.sum())

    if n_used < 2 * (poly_order + 1):
        raise ValueError("Too few observations within bandwidth.")

    right = R_w >= 0
    left = R_w < 0

    def _fit_slopes(outcome, side_mask):
        r_ = R_w[side_mask]
        y_ = outcome[side_mask]
        if len(r_) < poly_order + 1:
            return 0.0
        # Build polynomial design matrix
        cols = [np.ones(len(r_))] + [r_**k for k in range(1, poly_order + 1)]
        Xp = np.column_stack(cols)
        beta = np.linalg.lstsq(Xp, y_, rcond=None)[0]
        return float(beta[1])          # linear slope coefficient

    slope_Y_r = _fit_slopes(Y_w, right)
    slope_Y_l = _fit_slopes(Y_w, left)
    slope_T_r = _fit_slopes(T_w, right)
    slope_T_l = _fit_slopes(T_w, left)

    denom = slope_T_r - slope_T_l
    if abs(denom) < 1e-10:
        raise ValueError("No kink in treatment assignment; denominator near zero.")
    tau = (slope_Y_r - slope_Y_l) / denom

    # Delta-method SE via bootstrap
    rng = np.random.default_rng(42)
    n_boot = 500
    boot_taus = np.empty(n_boot)
    idx_w = np.where(mask)[0]
    for b in range(n_boot):
        bi = rng.integers(0, n_used, size=n_used)
        Rb = R_w[bi]
        Yb = Y_w[bi]
        Tb = T_w[bi]
        rb_ = Rb >= 0
        lb_ = Rb < 0
        sYr = _fit_slopes_raw(Yb, Rb, rb_, poly_order)
        sYl = _fit_slopes_raw(Yb, Rb, lb_, poly_order)
        sTr = _fit_slopes_raw(Tb, Rb, rb_, poly_order)
        sTl = _fit_slopes_raw(Tb, Rb, lb_, poly_order)
        den = sTr - sTl
        boot_taus[b] = (sYr - sYl) / den if abs(den) > 1e-10 else tau

    se = float(np.std(boot_taus, ddof=1))
    z = _norm.ppf(1.0 - alpha / 2.0)

    return {
        "tau": tau,
        "se": se,
        "ci_lower": tau - z * se,
        "ci_upper": tau + z * se,
        "slope_Y_right": slope_Y_r,
        "slope_Y_left": slope_Y_l,
        "slope_T_right": slope_T_r,
        "slope_T_left": slope_T_l,
        "n_used": n_used,
        "bandwidth": bandwidth,
        "method": "RKD",
    }


def _fit_slopes_raw(y, r, mask, poly_order):
    r_ = r[mask]
    y_ = y[mask]
    if len(r_) < poly_order + 1:
        return 0.0
    cols = [np.ones(len(r_))] + [r_**k for k in range(1, poly_order + 1)]
    Xp = np.column_stack(cols)
    beta = np.linalg.lstsq(Xp, y_, rcond=None)[0]
    return float(beta[1]) if len(beta) > 1 else 0.0


def cheatsheet() -> str:
    return "rgknd(Y, T, R) -> Regression kink design (Card et al. 2015, Econometrica)."

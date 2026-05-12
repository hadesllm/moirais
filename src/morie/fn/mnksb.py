# morie.fn -- function file (hadesllm/morie)
"""Manski bounds for partial identification of the ATE.

Under no-assumptions (worst-case) or monotone treatment response (MTR),
computes sharp bounds on the average treatment effect.

References
----------
Manski, C. F. (1990). Nonparametric bounds on treatment effects.
*American Economic Review*, 80(2), 319-323.

Manski, C. F. (1997). Monotone treatment response.
*Econometrica*, 65(6), 1311-1334.

Manski, C. F., & Pepper, J. V. (2000). Monotone instrumental
variables: With an application to the returns to schooling.
*Econometrica*, 68(4), 997-1010.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["mnksb"]

_Y_MIN_DEFAULT = 0.0
_Y_MAX_DEFAULT = 1.0


def mnksb(
    Y: np.ndarray,
    T: np.ndarray,
    *,
    y_min: float | None = None,
    y_max: float | None = None,
    assume_mtr: bool = False,
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Compute Manski no-assumptions (and optionally MTR) bounds on ATE.

    Under no assumptions, the ATE :math:`E[Y(1) - Y(0)]` lies in:

    .. math::

        \text{LB} &= E[Y|T=1]P(T=1) + y_{\min}P(T=0)
                    - E[Y|T=0]P(T=0) - y_{\max}P(T=1) \\
        \text{UB} &= E[Y|T=1]P(T=1) + y_{\max}P(T=0)
                    - E[Y|T=0]P(T=0) - y_{\min}P(T=1)

    Under monotone treatment response (MTR, :math:`Y(1) \geq Y(0)`),
    bounds are tightened to :math:`[0, \text{UB}]` intersected with
    :math:`[\text{LB}, y_{\max} - y_{\min}]`.

    Parameters
    ----------
    Y : np.ndarray
        Observed outcome, shape ``(n,)``.
    T : np.ndarray
        Binary treatment, shape ``(n,)``.
    y_min : float, optional
        Known minimum of the outcome support.
    y_max : float, optional
        Known maximum of the outcome support.
    assume_mtr : bool
        If True, impose monotone treatment response to tighten bounds.
    alpha : float
        Significance level for bootstrap confidence interval on bounds.

    Returns
    -------
    dict
        ``lb`` (lower bound), ``ub`` (upper bound), ``width``,
        ``lb_ci``, ``ub_ci``, ``n``, ``method``.

    References
    ----------
    Manski (1990). AER, 80(2), 319-323.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    if len(Y) != len(T):
        raise ValueError("Y and T must have the same length.")

    y0 = float(y_min) if y_min is not None else float(Y.min())
    y1 = float(y_max) if y_max is not None else float(Y.max())
    if y1 <= y0:
        raise ValueError("y_max must be strictly greater than y_min.")

    idx1 = T == 1
    idx0 = T == 0
    if idx1.sum() == 0 or idx0.sum() == 0:
        raise ValueError("Both treatment values must be present.")

    n = len(Y)
    p1 = float(idx1.sum()) / n
    p0 = 1.0 - p1
    mu1 = float(Y[idx1].mean())
    mu0 = float(Y[idx0].mean())

    lb, ub = _bounds(mu1, mu0, p1, p0, y0, y1, assume_mtr)

    # Bootstrap CIs on the bounds
    rng = np.random.default_rng(42)
    n_boot = 500
    lb_boots = np.empty(n_boot)
    ub_boots = np.empty(n_boot)
    for b in range(n_boot):
        bi = rng.integers(0, n, size=n)
        Yb, Tb = Y[bi], T[bi]
        i1, i0 = Tb == 1, Tb == 0
        if i1.sum() == 0 or i0.sum() == 0:
            lb_boots[b], ub_boots[b] = lb, ub
            continue
        p1b = i1.sum() / n
        p0b = 1 - p1b
        m1b, m0b = Yb[i1].mean(), Yb[i0].mean()
        lb_boots[b], ub_boots[b] = _bounds(m1b, m0b, p1b, p0b, y0, y1, assume_mtr)

    hw = _norm_ppf(1.0 - alpha / 2.0)

    return {
        "lb": lb,
        "ub": ub,
        "width": ub - lb,
        "lb_ci": (float(np.mean(lb_boots) - hw * np.std(lb_boots, ddof=1)),
                  float(np.mean(lb_boots) + hw * np.std(lb_boots, ddof=1))),
        "ub_ci": (float(np.mean(ub_boots) - hw * np.std(ub_boots, ddof=1)),
                  float(np.mean(ub_boots) + hw * np.std(ub_boots, ddof=1))),
        "n": n,
        "method": "Manski-bounds-MTR" if assume_mtr else "Manski-bounds",
    }


def _bounds(mu1, mu0, p1, p0, y0, y1, mtr):
    lb = mu1 * p1 + y0 * p0 - mu0 * p0 - y1 * p1
    ub = mu1 * p1 + y1 * p0 - mu0 * p0 - y0 * p1
    if mtr:
        lb = max(lb, 0.0)
        ub = min(ub, y1 - y0)
    return lb, ub


def _norm_ppf(q):
    from scipy.stats import norm
    return float(norm.ppf(q))


def cheatsheet() -> str:
    return "mnksb(Y, T) -> Manski no-assumptions bounds on ATE (Manski 1990, AER)."

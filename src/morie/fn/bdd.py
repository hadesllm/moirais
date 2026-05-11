# morie.fn — function file (hadesllm/morie)
"""Bunching difference-in-differences."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import ESRes
from ._helpers import _validate_df


def bunching_did(
    data: pd.DataFrame,
    *,
    r: str = "running",
    period: str = "post",
    cutoff: float = 0.0,
    bin_width: float | None = None,
    excluded_range: tuple[float, float] | None = None,
    poly_order: int = 5,
    alpha: float = 0.05,
) -> ESRes:
    r"""Bunching difference-in-differences estimator.

    Combines bunching estimation with a DiD design by comparing
    excess mass at the cutoff before and after a policy change:

    .. math::

        \hat{\Delta B} = \hat{B}_{post} - \hat{B}_{pre}

    where :math:`\hat{B}_t` is the bunching estimate in period t,
    constructed from polynomial counterfactual densities.

    Parameters
    ----------
    data : pd.DataFrame
    r : str
        Running variable column.
    period : str
        Binary pre/post indicator column.
    cutoff : float
        Threshold point.
    bin_width : float or None
        Histogram bin width.
    excluded_range : tuple or None
        Range around cutoff to exclude.
    poly_order : int
        Polynomial order for counterfactual.
    alpha : float
        Significance level.

    Returns
    -------
    ESRes
        estimate = change in excess bunching (post - pre).

    References
    ----------
    Kleven, H. J. (2016). Bunching. *Annual Review of Economics*, 8, 435-464.
    """
    _validate_df(data, r, period)
    df = data[[r, period]].dropna()
    R = df[r].to_numpy(dtype=float)
    P = df[period].to_numpy(dtype=float)
    n = len(R)

    if bin_width is None:
        iqr = float(np.percentile(R, 75) - np.percentile(R, 25))
        bin_width = 2 * iqr * n ** (-1 / 3) if iqr > 0 else R.std() * 0.1

    if excluded_range is None:
        excluded_range = (cutoff - 2 * bin_width, cutoff + 2 * bin_width)

    def _bunching_excess(r_sub):
        r_min = r_sub.min() - bin_width
        r_max = r_sub.max() + bin_width
        bins = np.arange(r_min, r_max + bin_width, bin_width)
        counts, edges = np.histogram(r_sub, bins=bins)
        mids = 0.5 * (edges[:-1] + edges[1:])
        excl = (mids >= excluded_range[0]) & (mids <= excluded_range[1])
        incl = ~excl
        if incl.sum() < poly_order + 1:
            return 0.0, 0.0
        z = (mids - cutoff) / (bin_width * 10)
        Xp = np.column_stack([z ** k for k in range(poly_order + 1)])
        beta = np.linalg.lstsq(Xp[incl], counts[incl], rcond=None)[0]
        cf = np.maximum(Xp @ beta, 0)
        excess = float(np.sum(counts[excl] - cf[excl]))
        resid = counts[incl] - Xp[incl] @ beta
        se_r = float(np.std(resid, ddof=min(poly_order + 1, len(resid) - 1)))
        se_b = se_r * np.sqrt(excl.sum())
        return excess, se_b

    b_pre, se_pre = _bunching_excess(R[P == 0])
    b_post, se_post = _bunching_excess(R[P == 1])

    delta = b_post - b_pre
    se = float(np.sqrt(se_pre ** 2 + se_post ** 2))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return ESRes(
        measure="Bunching DiD",
        estimate=delta,
        ci_lower=delta - z_crit * se,
        ci_upper=delta + z_crit * se,
        se=se,
        n=n,
        extra={
            "b_pre": b_pre,
            "b_post": b_post,
            "se_pre": se_pre,
            "se_post": se_post,
            "n_pre": int((P == 0).sum()),
            "n_post": int((P == 1).sum()),
        },
    )


bdd = bunching_did


def cheatsheet() -> str:
    return "bunching_did({}) -> Bunching difference-in-differences."

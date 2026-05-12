# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Bunching estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import ESRes
from ._helpers import _validate_df


def bunching(
    data: pd.DataFrame,
    *,
    r: str = "running",
    cutoff: float = 0.0,
    bin_width: float | None = None,
    excluded_range: tuple[float, float] | None = None,
    poly_order: int = 7,
) -> ESRes:
    r"""Bunching estimator for behavioral responses at a threshold.

    Estimates excess mass at the cutoff by comparing the observed
    distribution to a counterfactual polynomial fit:

    .. math::

        c_j = \sum_{k=0}^{p} \beta_k (z_j)^k
        + \sum_{j \in \text{excluded}} \gamma_j D_j + \epsilon_j

    The excess bunching is:

    .. math::

        \hat{B} = \sum_{j \in \text{excluded}} (c_j^{obs} - \hat{c}_j^{cf})

    and the elasticity estimate is:

    .. math::

        \hat{e} = \frac{\hat{B} / \bar{h}}{z^* \cdot t / (1-t)}

    Parameters
    ----------
    data : pd.DataFrame
    r : str
        Running variable column.
    cutoff : float
        Threshold/kink/notch point.
    bin_width : float or None
        Histogram bin width. If None, uses Freedman-Diaconis.
    excluded_range : tuple or None
        Range around cutoff to exclude from counterfactual estimation.
    poly_order : int
        Polynomial order for counterfactual density.

    Returns
    -------
    ESRes
        estimate = excess mass (bunching count).

    References
    ----------
    Kleven, H. J. (2016). Bunching. *Annual Review of Economics*, 8, 435-464.
    Saez, E. (2010). Do taxpayers bunch at kink points? *AEJ: Econ Policy*.
    """
    _validate_df(data, r)
    R = data[r].dropna().to_numpy(dtype=float)
    n = len(R)

    if bin_width is None:
        iqr = float(np.percentile(R, 75) - np.percentile(R, 25))
        bin_width = 2 * iqr * n ** (-1 / 3) if iqr > 0 else R.std() * 0.1

    if excluded_range is None:
        excluded_range = (cutoff - 2 * bin_width, cutoff + 2 * bin_width)

    r_min = R.min() - bin_width
    r_max = R.max() + bin_width
    bins = np.arange(r_min, r_max + bin_width, bin_width)
    counts, edges = np.histogram(R, bins=bins)
    midpoints = 0.5 * (edges[:-1] + edges[1:])

    excluded_mask = ((midpoints >= excluded_range[0]) &
                     (midpoints <= excluded_range[1]))
    included_mask = ~excluded_mask

    if included_mask.sum() < poly_order + 1:
        raise ValueError("Too few bins outside excluded range")

    z_norm = (midpoints - cutoff) / (bin_width * 10)
    X_poly = np.column_stack([z_norm ** k for k in range(poly_order + 1)])

    X_incl = X_poly[included_mask]
    c_incl = counts[included_mask]
    beta = np.linalg.lstsq(X_incl, c_incl, rcond=None)[0]

    counterfactual = X_poly @ beta
    counterfactual = np.maximum(counterfactual, 0)

    excess = float(np.sum(counts[excluded_mask] - counterfactual[excluded_mask]))

    resid = c_incl - X_incl @ beta
    se_resid = float(np.std(resid, ddof=poly_order + 1))
    n_excl = int(excluded_mask.sum())
    se_bunch = se_resid * np.sqrt(n_excl)

    return ESRes(
        measure="Bunching excess mass",
        estimate=excess,
        ci_lower=excess - 1.96 * se_bunch,
        ci_upper=excess + 1.96 * se_bunch,
        se=se_bunch,
        n=n,
        extra={
            "n_bins": len(counts),
            "n_excluded_bins": n_excl,
            "bin_width": bin_width,
            "excluded_range": excluded_range,
            "poly_order": poly_order,
        },
    )


bunck = bunching


def cheatsheet() -> str:
    return "bunching({}) -> Bunching estimator."

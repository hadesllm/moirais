"""Normal score transform."""

from __future__ import annotations

import numpy as np

from ._containers import SpatialResult


def normal_score_transform(Z: np.ndarray) -> SpatialResult:
    r"""Transform data to Gaussian quantiles (normal scores).

    .. math::

        Y_i = \Phi^{-1}\!\left(\frac{R_i - 0.5}{n}\right)

    where :math:`R_i` is the rank of :math:`Z_i`.

    Parameters
    ----------
    Z : np.ndarray
        Observed values, shape ``(n,)``.

    Returns
    -------
    SpatialResult
        ``statistic`` is the Shapiro-Wilk W of the transformed data.
        ``extra`` has ``transformed``, ``ranks``.

    References
    ----------
    Schabenberger & Gotway (2005), Ch. 5.

    .. epigraph::

        "Had to be me. Someone else might have gotten it wrong."
        -- Mordin, Mass Effect
    """
    from scipy.stats import norm, shapiro

    Z = np.asarray(Z, dtype=np.float64).ravel()
    n = len(Z)
    ranks = np.argsort(np.argsort(Z)).astype(np.float64)
    Y = norm.ppf((ranks + 0.5) / n)
    w_stat = float(shapiro(Y).statistic) if n >= 3 else 1.0

    return SpatialResult(
        name="normal_score_transform",
        statistic=w_stat,
        p_value=None,
        extra={"transformed": Y, "ranks": ranks},
    )


sgnsc = normal_score_transform


def cheatsheet() -> str:
    return "normal_score_transform({}) -> Normal score transform."

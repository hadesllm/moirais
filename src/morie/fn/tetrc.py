"""Tetrachoric correlation."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import brentq

from ._containers import ESRes


def tetrachoric_corr(a: int, b: int, c: int, d: int) -> ESRes:
    r"""Tetrachoric correlation from a 2x2 table.

    Table layout::

        |     | Y=1 | Y=0 |
        | X=1 |  a  |  b  |
        | X=0 |  c  |  d  |

    Parameters
    ----------
    a, b, c, d : int
        Cell counts.

    Returns
    -------
    ESRes
    """
    n = a + b + c + d
    if n < 1:
        raise ValueError("Total count must be >= 1.")

    ad = a * d
    bc = b * c
    if bc == 0 and ad == 0:
        return ESRes(measure="tetrachoric_r", estimate=0.0, n=n)

    if bc == 0:
        return ESRes(measure="tetrachoric_r", estimate=1.0, n=n)
    if ad == 0:
        return ESRes(measure="tetrachoric_r", estimate=-1.0, n=n)

    or_val = ad / bc
    r_approx = np.cos(np.pi / (1 + np.sqrt(or_val)))

    try:
        h1 = sp_stats.norm.ppf((a + b) / n)
        k1 = sp_stats.norm.ppf((a + c) / n)

        def _eq(rho):
            from scipy.stats import mvn

            lo = np.array([-np.inf, -np.inf])
            hi = np.array([h1, k1])
            cov = np.array([[1.0, rho], [rho, 1.0]])
            p, _ = mvn.mvnun(lo, hi, np.zeros(2), cov)
            return p - a / n

        r_tet = brentq(_eq, -0.999, 0.999)
    except Exception:
        r_tet = r_approx

    se = 1.0 / np.sqrt(n) if n > 0 else 0.0

    return ESRes(
        measure="tetrachoric_r",
        estimate=float(r_tet),
        se=float(se),
        n=n,
        extra={"odds_ratio": float(or_val), "approx_cospi": float(r_approx)},
    )


tetrc = tetrachoric_corr


def cheatsheet() -> str:
    return "tetrachoric_corr({}) -> Tetrachoric correlation."

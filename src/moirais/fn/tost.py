"""TOST equivalence test — two one-sided t-tests."""

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def tost_test(x, y, margin=0.5, cdf=None):
    """
    Two One-Sided Tests (TOST) for equivalence.

    Tests H0: |mu_x - mu_y| >= margin vs H1: |mu_x - mu_y| < margin.

    :param x: (n,) first sample.
    :param y: (m,) second sample.
    :param margin: Equivalence margin (positive).
    :return: DescriptiveResult with TOST p-value, CIs.

    References
    ----------
    Schuirmann DJ (1987). A Comparison of the Two One-Sided Tests Procedure
    and the Power Approach for Assessing the Equivalence of Average
    Bioavailability. J Pharmacokinetics Biopharmaceutics 15(6):657-680.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    margin = abs(margin)
    diff = x.mean() - y.mean()
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    se = sp * np.sqrt(1 / nx + 1 / ny)
    df = nx + ny - 2

    t_lower = (diff - (-margin)) / se if se > 0 else np.inf
    t_upper = (diff - margin) / se if se > 0 else -np.inf
    p_lower = stats.t.sf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)
    equivalent = p_tost < 0.05

    ci_low = diff - stats.t.ppf(0.975, df) * se
    ci_high = diff + stats.t.ppf(0.975, df) * se

    return DescriptiveResult(
        name="tost_test",
        value=float(p_tost),
        extra={
            "p_value": float(p_tost),
            "p_lower": float(p_lower),
            "p_upper": float(p_upper),
            "mean_diff": float(diff),
            "margin": float(margin),
            "ci_90": [float(ci_low), float(ci_high)],
            "equivalent": bool(equivalent),
            "n_x": nx,
            "n_y": ny,
        },
    )


def cheatsheet() -> str:
    return "tost_test({}) -> TOST equivalence test — two one-sided t-tests."

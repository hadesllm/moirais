# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Brunner-Munzel test for stochastic equality."""

import numpy as np
from scipy import stats

from ._containers import DescriptiveResult


def brunner_munzel(x, y):
    """
    Brunner-Munzel test for stochastic equality.

    Tests H0: P(X < Y) + 0.5*P(X=Y) = 0.5 (stochastic equality).
    More robust than Wilcoxon for heteroscedastic data.

    :param x: (n,) first sample.
    :param y: (m,) second sample.
    :return: DescriptiveResult with test statistic, p-value, effect size.

    References
    ----------
    Brunner E, Munzel U (2000). The Nonparametric Behrens-Fisher Problem.
    Biometrical Journal 42(1):17-25.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    nx, ny = len(x), len(y)
    N = nx + ny

    combined = np.concatenate([x, y])
    ranks_all = stats.rankdata(combined)
    ranks_x_in_all = ranks_all[:nx]
    ranks_y_in_all = ranks_all[nx:]

    mean_rx = ranks_x_in_all.mean()
    mean_ry = ranks_y_in_all.mean()

    ranks_x = stats.rankdata(x)
    ranks_y = stats.rankdata(y)
    sx2 = np.var(ranks_x_in_all - ranks_x, ddof=1)
    sy2 = np.var(ranks_y_in_all - ranks_y, ddof=1)

    p_hat = (mean_ry - mean_rx) / N + 0.5
    denom = np.sqrt(N * (sx2 / nx + sy2 / ny))
    w = N * (mean_ry - mean_rx) / denom if denom > 0 else 0.0

    f_val = (sx2 / nx + sy2 / ny) ** 2
    f_num = f_val
    f_den = (sx2 / nx) ** 2 / (nx - 1) + (sy2 / ny) ** 2 / (ny - 1)
    df = f_num / f_den if f_den > 0 else nx + ny - 2
    p_value = 2 * stats.t.sf(abs(w), df)

    return DescriptiveResult(
        name="brunner_munzel",
        value=float(w),
        extra={
            "statistic": float(w),
            "p_value": float(p_value),
            "p_hat": float(p_hat),
            "df": float(df),
            "n_x": nx,
            "n_y": ny,
        },
    )


def cheatsheet() -> str:
    return "brunner_munzel({}) -> Brunner-Munzel test for stochastic equality."

# morie.fn -- function file (hadesllm/morie)
"""Neyman-orthogonal score computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import ESRes
from ._helpers import _validate_df


def orthogonal_score(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    m_hat: str = "m_hat",
    e_hat: str = "e_hat",
    alpha: float = 0.05,
) -> ESRes:
    r"""Compute Neyman-orthogonal score for partially linear model.

    Given pre-computed nuisance predictions :math:`\hat{m}(X) = E[Y|X]`
    and :math:`\hat{e}(X) = E[T|X]`, the orthogonal score is:

    .. math::

        \psi_i = \hat{V}_i (Y_i - \hat{m}(X_i) - \hat{\theta} \hat{V}_i)

    where :math:`\hat{V}_i = T_i - \hat{e}(X_i)` and

    .. math::

        \hat{\theta} = \frac{\sum_i \hat{V}_i (Y_i - \hat{m}(X_i))}
        {\sum_i \hat{V}_i^2}

    The orthogonality property ensures :math:`\sqrt{n}`-consistency
    even when nuisance functions converge at slower rates.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    t : str
        Treatment column.
    m_hat : str
        Column with E[Y|X] predictions.
    e_hat : str
        Column with E[T|X] (or P(T=1|X)) predictions.
    alpha : float
        Significance level.

    Returns
    -------
    ESRes

    References
    ----------
    Neyman, J. (1959). Optimal asymptotic tests of composite hypotheses.
    *Probability and Statistics*, 213-234.
    Chernozhukov, V., et al. (2018). Double/debiased machine learning.
    *The Econometrics Journal*, 21(1), C1-C68.
    """
    _validate_df(data, y, t, m_hat, e_hat)
    df = data[[y, t, m_hat, e_hat]].dropna()
    n = len(df)

    Y = df[y].to_numpy(dtype=float)
    T_arr = df[t].to_numpy(dtype=float)
    M = df[m_hat].to_numpy(dtype=float)
    E = df[e_hat].to_numpy(dtype=float)

    V = T_arr - E
    Y_tilde = Y - M

    denom = float(np.sum(V ** 2))
    if denom < 1e-15:
        raise ValueError("Degenerate: V = T - e_hat is near zero for all obs")

    theta = float(np.sum(V * Y_tilde) / denom)

    scores = V * (Y_tilde - theta * V) / (denom / n)
    se = float(np.std(scores, ddof=1) / np.sqrt(n))

    z = stats.norm.ppf(1 - alpha / 2)
    return ESRes(
        measure="Neyman-orthogonal ATE",
        estimate=theta,
        ci_lower=theta - z * se,
        ci_upper=theta + z * se,
        se=se,
        n=n,
        extra={"mean_score": float(np.mean(scores))},
    )


ortho = orthogonal_score


def cheatsheet() -> str:
    return "orthogonal_score({}) -> Neyman-orthogonal score computation."

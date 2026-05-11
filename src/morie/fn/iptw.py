# morie.fn — function file (hadesllm/morie)
"""Stabilized IPTW with diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import ESRes
from ._helpers import _validate_df


def stabilized_iptw(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    covariates: list[str] | None = None,
    ps_col: str | None = None,
    trim: tuple[float, float] = (0.01, 0.99),
    alpha: float = 0.05,
) -> ESRes:
    r"""Stabilized inverse probability of treatment weighting.

    Stabilized weights reduce variance by multiplying by marginal
    treatment probability:

    .. math::

        sw_i = \begin{cases}
        \frac{P(T=1)}{e(X_i)} & \text{if } T_i=1 \\
        \frac{P(T=0)}{1 - e(X_i)} & \text{if } T_i=0
        \end{cases}

    The ATE is then:

    .. math::

        \hat{\tau}_{IPTW} = \frac{1}{n}\sum_{i=1}^{n}
        \left[\frac{T_i Y_i}{e(X_i)} P(T=1)
        - \frac{(1-T_i) Y_i}{1-e(X_i)} P(T=0)\right]

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    t : str
        Binary treatment column.
    covariates : list[str] or None
        Columns for PS estimation (if ps_col is None).
    ps_col : str or None
        Pre-computed propensity score column.
    trim : tuple
        Quantile bounds for trimming propensity scores.
    alpha : float
        Significance level.

    Returns
    -------
    ESRes

    References
    ----------
    Robins, J. M., Hernan, M. A., & Brumback, B. (2000). Marginal structural
    models and causal inference in epidemiology. *Epidemiology*, 11(5), 550-560.
    """
    _validate_df(data, y, t)
    df = data.dropna(subset=[y, t])
    T = df[t].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)

    if ps_col is not None:
        ps = df[ps_col].to_numpy(dtype=float)
    else:
        if covariates is None or len(covariates) == 0:
            raise ValueError("Provide covariates or ps_col")
        _validate_df(df, *covariates)
        X = df[covariates].to_numpy(dtype=float)
        X_design = np.column_stack([np.ones(len(X)), X])
        beta = np.zeros(X_design.shape[1])
        for _ in range(50):
            p = 1.0 / (1.0 + np.exp(-np.clip(X_design @ beta, -500, 500)))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            W = p * (1 - p)
            grad = X_design.T @ (T - p)
            H = X_design.T @ (X_design * W[:, None])
            try:
                beta += np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
        ps = 1.0 / (1.0 + np.exp(-np.clip(X_design @ beta, -500, 500)))

    ps = np.clip(ps, trim[0], trim[1])

    p_treat = T.mean()
    sw = np.where(T == 1, p_treat / ps, (1 - p_treat) / (1 - ps))

    ate = float(np.mean(sw * T * Y) / np.mean(sw * T)
                - np.mean(sw * (1 - T) * Y) / np.mean(sw * (1 - T)))

    scores = sw * T * (Y - ate) / np.mean(sw * T) - sw * (1 - T) * Y / np.mean(sw * (1 - T))
    se = float(np.std(scores, ddof=1) / np.sqrt(len(T)))

    z = stats.norm.ppf(1 - alpha / 2)
    ess_t = float(np.sum(sw[T == 1]) ** 2 / np.sum(sw[T == 1] ** 2))
    ess_c = float(np.sum(sw[T == 0]) ** 2 / np.sum(sw[T == 0] ** 2))

    return ESRes(
        measure="Stabilized IPTW ATE",
        estimate=ate,
        ci_lower=ate - z * se,
        ci_upper=ate + z * se,
        se=se,
        n=len(T),
        extra={
            "ess_treated": ess_t,
            "ess_control": ess_c,
            "max_weight": float(sw.max()),
            "mean_ps": float(ps.mean()),
        },
    )


iptw = stabilized_iptw


def cheatsheet() -> str:
    return "stabilized_iptw({}) -> Stabilized IPTW with diagnostics."

# moirais.fn — function file (hadesllm/moirais)
"""Propensity score matching. 'This is the Way.' -- The Mandalorian"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from ._containers import ESRes
from ._helpers import _validate_df


def ps_match(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    t: str = "treatment",
    x: list[str] | str = "x",
    caliper: float = 0.2,
) -> ESRes:
    """Propensity score nearest-neighbor matching (1:1, without replacement)."""
    _validate_df(data, y, t)
    if isinstance(x, str):
        x = [x]
    _validate_df(data, *x)
    df = data[[y, t, *x]].dropna().reset_index(drop=True)

    # Estimate propensity via simple logistic (IRLS, 20 iterations)
    Y_t = df[t].to_numpy(dtype=float)
    X_cov = np.column_stack([np.ones(len(df)), df[x].to_numpy(dtype=float)])
    beta = np.zeros(X_cov.shape[1])
    for _ in range(20):
        eta = X_cov @ beta
        mu = expit(np.clip(eta, -20, 20))
        mu = np.clip(mu, 1e-8, 1 - 1e-8)
        W = mu * (1 - mu)
        z = eta + (Y_t - mu) / W
        try:
            beta = np.linalg.solve(X_cov.T @ (X_cov * W[:, None]), X_cov.T @ (W * z))
        except np.linalg.LinAlgError:
            break
    ps = expit(np.clip(X_cov @ beta, -20, 20))

    # Match treated to nearest control
    treated_idx = np.where(Y_t == 1)[0]
    control_idx = np.where(Y_t == 0)[0]
    ps_t = ps[treated_idx]
    ps_c = ps[control_idx]
    matched_t, matched_c = [], []
    used: set[int] = set()
    for i, psi in enumerate(ps_t):
        dists = np.abs(ps_c - psi)
        for j in np.argsort(dists):
            if j not in used and dists[j] <= caliper * ps.std():
                matched_t.append(treated_idx[i])
                matched_c.append(control_idx[j])
                used.add(j)
                break
    if len(matched_t) == 0:
        raise ValueError("No matches found within caliper")

    y_mt = df[y].iloc[matched_t].to_numpy(dtype=float)
    y_mc = df[y].iloc[matched_c].to_numpy(dtype=float)
    ate = float(y_mt.mean() - y_mc.mean())
    se = float(np.sqrt(y_mt.var(ddof=1) / len(y_mt) + y_mc.var(ddof=1) / len(y_mc)))
    z_crit = stats.norm.ppf(0.975)
    return ESRes(
        measure="ATT (PS matching)",
        estimate=ate,
        ci_lower=ate - z_crit * se,
        ci_upper=ate + z_crit * se,
        se=se,
        n=len(matched_t) * 2,
        extra={"n_matched": len(matched_t), "caliper": caliper},
    )


mando = ps_match


def cheatsheet() -> str:
    return "ps_match({}) -> Propensity score matching. 'This is the Way.' -- The Mandalo"

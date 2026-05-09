"""You have power over your mind — not outside events. — Marcus Aurelius"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from ._containers import ESRes
from ._helpers import _validate_df


def synth_control(
    data: pd.DataFrame,
    *,
    y: str = "outcome",
    unit: str = "unit",
    time: str = "time",
    treated_unit: str = "treated",
    treat_time: int = 0,
) -> ESRes:
    """Synthetic control method: weighted combination of control units to match treated."""
    _validate_df(data, y, unit, time)
    pre = data[data[time] < treat_time]
    post = data[data[time] >= treat_time]

    treated_pre = pre[pre[unit] == treated_unit][y].to_numpy(dtype=float)
    control_units = [u for u in data[unit].unique() if u != treated_unit]
    if len(control_units) == 0:
        raise ValueError("No control units found")

    C = np.column_stack([pre[pre[unit] == u][y].to_numpy(dtype=float) for u in control_units])
    n_ctrl = len(control_units)

    # Find weights that minimize pre-period MSE
    def obj(w):
        synth = C @ w
        min_len = min(len(synth), len(treated_pre))
        return float(np.sum((treated_pre[:min_len] - synth[:min_len]) ** 2))

    # Constrained: weights sum to 1, non-negative
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_ctrl
    w0 = np.ones(n_ctrl) / n_ctrl
    res = optimize.minimize(obj, w0, bounds=bounds, constraints=cons, method="SLSQP")
    weights = res.x

    # Post-period: treated - synthetic
    treated_post = post[post[unit] == treated_unit][y].to_numpy(dtype=float)
    C_post = np.column_stack([post[post[unit] == u][y].to_numpy(dtype=float) for u in control_units])
    synth_post = C_post @ weights
    min_len = min(len(treated_post), len(synth_post))
    effect = float(np.mean(treated_post[:min_len] - synth_post[:min_len]))

    return ESRes(
        measure="Synthetic control ATT",
        estimate=effect,
        n=min_len,
        extra={
            "weights": dict(zip(control_units, weights.tolist())),
            "pre_rmse"All models are wrong, but some are useful. — George E. P. Box"synth_control({}) -> Synthetic control. 'Peace is a lie, there is only passion.' "

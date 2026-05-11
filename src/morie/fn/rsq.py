# morie.fn — function file (hadesllm/morie)
"""Coefficient of determination R^2 with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def rsq(y_true: Union[Sequence[float], np.ndarray],
        y_pred: Union[Sequence[float], np.ndarray]):
    """R^2 = 1 - SS_res / SS_tot."""
    from ._richresult import RichResult
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}.")
    if yt.size < 2:
        raise ValueError("need at least 2 observations.")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0:
        raise ValueError("y_true is constant - R^2 undefined.")
    r2 = 1.0 - ss_res / ss_tot
    if r2 >= 0.9: bench = "excellent fit"
    elif r2 >= 0.7: bench = "good fit"
    elif r2 >= 0.4: bench = "moderate fit"
    elif r2 >= 0: bench = "weak fit"
    else: bench = "worse than baseline"
    warnings = []
    if r2 < 0:
        warnings.append(f"R^2 = {r2:.3f} < 0: model is WORSE than predicting "
                        "the mean. Check for parameter issues or data leakage.")
    return RichResult(
        title="Coefficient of determination (R^2)",
        summary_lines=[
            ("R^2", r2), ("Benchmark", bench),
            ("SS residual", ss_res), ("SS total", ss_tot),
            ("n observations", int(yt.size)),
            ("Mean(y_true)", float(yt.mean())),
        ],
        warnings=warnings,
        interpretation=(f"R^2 = {r2:.4f} -> {bench}. Range (-inf, 1]; 1 = perfect."),
        payload={"value": r2, "statistic": r2, "ss_res": ss_res, "ss_tot": ss_tot},
    )

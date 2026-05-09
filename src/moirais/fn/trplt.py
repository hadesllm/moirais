"""Trace plot data generation."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def trace_plot_data(
    samples: Union[list, np.ndarray],
    *,
    param_names: Union[list, None] = None,
) -> dict[str, Any]:
    """
    Generate data for MCMC trace plots.

    :param samples: MCMC samples. Shape (n_iter,) for 1-D or (n_iter, d) for multi-param.
    :param param_names: Optional parameter names.
    :return: Dictionary with iteration indices, traces, running means per parameter.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n_iter, d = arr.shape

    if param_names is None:
        param_names = [f"param_{j}" for j in range(d)]

    traces = {}
    for j in range(d):
        col = arr[:, j]
        cumsum = np.cumsum(col)
        running_mean = cumsum / np.arange(1, n_iter + 1)
        traces[param_names[j]] = {
            "values": col.tolist(),
            "running_mean": running_mean.tolist(),
            "mean": float(np.mean(col)),
            "sd": float(np.std(col, ddof=1)) if n_iter > 1 else 0.0,
        }

    return {
        "iterations": list(range(n_iter)),
        "traces": traces,
        "n_iter": n_iter,
        "n_params": d,
    }


trplt = trace_plot_data


def cheatsheet() -> str:
    return "trace_plot_data({}) -> Trace plot data generation."

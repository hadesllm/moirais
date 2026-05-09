# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bidomain cardiac propagation model."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bidomain_model_fn(
    n_cells: int = 50,
    duration: float = 10.0,
    dt: float = 0.1,
    conductivity: float = 1.0,
) -> DescriptiveResult:
    """Simulate cardiac action potential propagation via bidomain model.

    :param n_cells: Number of cells in 1-D chain (default 50).
    :param duration: Simulation duration in ms (default 10).
    :param dt: Time step in ms (default 0.1).
    :param conductivity: Inter-cell conductivity (default 1.0).
    :return: DescriptiveResult with voltage matrix in extra.
    """
    from moirais._biomodel import bidomain_model

    t, V = bidomain_model(
        n_cells=n_cells,
        duration=duration,
        dt=dt,
        conductivity=conductivity,
    )
    return DescriptiveResult(
        name="bidomain",
        value=n_cells,
        extra={
            "time": t,
            "voltage_matrix": V,
            "n_cells": n_cells,
            "conductivity": conductivity,
        },
    )


bidmn = bidomain_model_fn


def cheatsheet() -> str:
    return "bidomain_model_fn({}) -> Bidomain cardiac propagation model."

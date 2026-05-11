"""Trace plot data for MCMC."""

from __future__ import annotations

from ._containers import DescriptiveResult


def trace_plot_data(chain) -> DescriptiveResult:
    """Extract trace data for MCMC diagnostics.

    .. epigraph:: "Hodor!" -- Hodor, Game of Thrones
    """
    import numpy as np

    chain = np.asarray(chain, dtype=float)
    if chain.ndim == 1:
        chain = chain.reshape(-1, 1)
    n_samples, n_params = chain.shape
    traces = {f"param_{i}": chain[:, i].tolist() for i in range(n_params)}
    return DescriptiveResult(
        name="trace_plot_data",
        value=float(n_samples),
        extra={
            "traces": traces,
            "n_samples": n_samples,
            "n_params": n_params,
            "iterations": list(range(n_samples)),
        },
    )


trcpl = trace_plot_data


def cheatsheet() -> str:
    return "trace_plot_data({}) -> Trace plot data for MCMC."

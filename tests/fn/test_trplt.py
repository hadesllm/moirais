"""Tests for morie.fn.trplt -- Trace plot data."""

import numpy as np
from morie.fn.trplt import trace_plot_data


def test_returns_dict():
    samples = np.random.default_rng(42).standard_normal(100)
    result = trace_plot_data(samples)
    assert isinstance(result, dict)
    assert "traces" in result


def test_running_mean_length():
    samples = np.random.default_rng(42).standard_normal((200, 2))
    result = trace_plot_data(samples, param_names=["a", "b"])
    assert len(result["traces"]["a"]["running_mean"]) == 200


def test_n_params():
    samples = np.random.default_rng(42).standard_normal((100, 3))
    result = trace_plot_data(samples)
    assert result["n_params"] == 3

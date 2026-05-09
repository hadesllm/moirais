"""Tests for moirais.fn.trcpl -- trace plot data."""
import numpy as np
from moirais.fn.trcpl import trace_plot_data, trcpl


def test_alias():
    assert trcpl is trace_plot_data


def test_smoke():
    chain = np.random.default_rng(42).standard_normal((100, 3))
    r = trace_plot_data(chain)
    assert r.name == "trace_plot_data"
    assert r.extra["n_params"] == 3
    assert r.extra["n_samples"] == 100
    assert "param_0" in r.extra["traces"]

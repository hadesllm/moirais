"""Tests for fn/btsmp.py -- Bootstrap resampling."""
import numpy as np
import pandas as pd

from morie.fn.btsmp import btsmp, bootstrap_sample


def test_btsmp_returns_dict():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.normal(5, 1, size=100)})
    result = btsmp(df, 200, statistic=lambda d: d["x"].mean(), seed=42)
    assert isinstance(result, dict)
    assert "mean" in result
    assert "se" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "distribution" in result


def test_btsmp_mean_close_to_true():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.normal(10, 1, size=500)})
    result = bootstrap_sample(df, 500, statistic=lambda d: d["x"].mean(), seed=42)
    assert abs(result["mean"] - 10.0) < 0.5


def test_btsmp_ci_width():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.normal(0, 1, size=200)})
    result = btsmp(df, 300, statistic=lambda d: d["x"].mean(), seed=42)
    ci_width = result["ci_upper"] - result["ci_lower"]
    assert 0 < ci_width < 1.0  # Reasonable CI width for n=200

"""Tests for fn/jksmp.py -- Jackknife variance estimation."""
import numpy as np
import pandas as pd

from moirais.fn.jksmp import jksmp, jackknife_estimate


def test_jksmp_returns_dict():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = jksmp(df, statistic=lambda d: d["x"].mean())
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result
    assert "bias" in result


def test_jksmp_mean_exact():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = jackknife_estimate(df, statistic=lambda d: d["x"].mean())
    assert abs(result["estimate"] - 3.0) < 1e-10


def test_jksmp_se_positive():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.normal(0, 1, size=20)})
    result = jksmp(df, statistic=lambda d: d["x"].mean())
    assert result["se"] > 0

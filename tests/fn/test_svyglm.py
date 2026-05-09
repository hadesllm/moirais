"""Tests for fn/svyglm.py -- Complex survey GLM."""
import numpy as np
import pandas as pd
import pytest

from moirais.fn.svyglm import svyglm, complex_survey_glm


def test_svyglm_gaussian():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(0, 1, size=n)
    y = 2.0 * x + rng.normal(0, 0.5, size=n)
    df = pd.DataFrame({
        "y": y,
        "x": x,
        "w": rng.uniform(1, 5, size=n),
    })
    result = svyglm(df, "y ~ x", "w", family="gaussian")
    # Coefficient for x should be close to 2.0
    assert abs(result.params["x"] - 2.0) < 0.5


def test_svyglm_binomial():
    rng = np.random.default_rng(42)
    n = 300
    x = rng.normal(0, 1, size=n)
    prob = 1 / (1 + np.exp(-x))
    y = rng.binomial(1, prob, size=n)
    df = pd.DataFrame({
        "y": y,
        "x": x,
        "w": rng.uniform(1, 3, size=n),
    })
    result = complex_survey_glm(df, "y ~ x", "w", family="binomial")
    assert hasattr(result, "params")
    assert "x" in result.params.index


def test_svyglm_missing_weight_col():
    df = pd.DataFrame({"y": [1, 2], "x": [0, 1]})
    with pytest.raises(ValueError, match="Weight column"):
        svyglm(df, "y ~ x", "missing_col")

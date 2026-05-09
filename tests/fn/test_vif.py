"""Tests for moirais.fn.vif — Variance Inflation Factor."""
import numpy as np
import pandas as pd
import pytest

from moirais.fn.vif import variance_inflation, vif


def test_uncorrelated_vif_near_one():
    """Uncorrelated predictors should have VIF near 1."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"a": rng.standard_normal(200), "b": rng.standard_normal(200),
                        "c": rng.standard_normal(200)})
    result = variance_inflation(df)
    for v in result.value.values():
        assert v < 2.0, f"VIF={v} too high for uncorrelated predictors"


def test_correlated_vif_high():
    """Highly correlated predictors should have VIF >> 1."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    df = pd.DataFrame({"a": x, "b": x + rng.standard_normal(200) * 0.1})
    result = variance_inflation(df)
    assert result.extra["max_vif"] > 5


def test_vif_alias():
    assert vif is variance_inflation


def test_too_few_predictors():
    with pytest.raises(ValueError):
        variance_inflation(pd.DataFrame({"a": [1, 2, 3]}))

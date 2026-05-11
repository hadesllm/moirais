"""Tests for stexp."""
import numpy as np
import pytest
from morie.fn.stexp import stexp


def test_stexp_basic():
    result = stexp()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-ProductSum"


def test_stexp_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stexp(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stexp_no_data():
    result = stexp(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stexp_alias():
    from morie.fn.stexp import stexp
    assert stexp is stexp

"""Tests for stmet."""
import numpy as np
import pytest
from moirais.fn.stmet import stmet


def test_stmet_basic():
    result = stmet()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-IntegratedMetric"


def test_stmet_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stmet(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stmet_no_data():
    result = stmet(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stmet_alias():
    from moirais.fn.stmet import stmet
    assert stmet is stmet

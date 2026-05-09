"""Tests for stpor."""
import numpy as np
import pytest
from moirais.fn.stpor import stpor


def test_stpor_basic():
    result = stpor()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "ST-Porcu"


def test_stpor_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = stpor(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_stpor_no_data():
    result = stpor(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_stpor_alias():
    from moirais.fn.stpor import stpor
    assert stpor is stpor

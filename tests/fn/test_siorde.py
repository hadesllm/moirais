"""Tests for siorde."""
import numpy as np
import pytest
from moirais.fn.siorde import siorde


def test_siorde_basic():
    result = siorde()
    assert hasattr(result, "statistic")
    assert isinstance(result.statistic, float)
    assert result.name == "SIS-OrderRelation"


def test_siorde_with_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20)
    coords = rng.uniform(0, 1, size=(20, 2))
    result = siorde(data=data, coords=coords, n=20, seed=0)
    assert result.statistic == pytest.approx(float(np.mean(data)))
    assert result.extra["n_points"] == 20


def test_siorde_no_data():
    result = siorde(n=50, seed=7)
    assert result.statistic is not None
    assert result.extra["n_points"] == 50


def test_siorde_alias():
    from moirais.fn.siorde import siorde
    assert siorde is siorde
